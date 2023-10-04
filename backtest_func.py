import pandas as pd
import numpy as np
from paneldb import *
import time
from copy import deepcopy
from preprocessClass import *
from preprocess import*
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import os,sys
import functools

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False 

pdb = PanelDB()
pdb.db_connect('original_database')

#%% fuction
def clock(func):
    @functools.wraps(func)
    def clocked(*args,**kwargs):
        t0 = time.time()
        result = func(*args,**kwargs)
        elasped = time.time() - t0
        name = func.__name__
        print('[%0.8fs]-> %s' % (elasped,name))
        return result
    return clocked

@clock
#%% 获取基础数据
def get_stock_data(start, end):
    # open price计算收益率
    pdb.get_history(start, end)
    history0 = pdb.history
    open_df_per = pdb.get_price('open')
    close_df_per= pdb.get_price('close')
    high_df_per= pdb.get_price('high')
    limitup_per = pdb.get_price('limit_up')
    limitdown_per = pdb.get_price('limit_down')
    low_df_per= pdb.get_price('low')
    # 成交量数据
    vol = pdb.get_price('volume')
    # 一字涨停
    ztyizi_open_per = limitup_per == low_df_per
    # 一字跌停
    dtyizi_open_per = limitdown_per == high_df_per
    # 停牌
    tingpai_per = vol==0
    #true fasle转换成1，0
    tingpai_per = tingpai_per.astype(int) 
    ztyizi_open_per = ztyizi_open_per.astype(int)
    dtyizi_open_per = dtyizi_open_per.astype(int) 
    # st矩阵 
    st_id = limitup_per/limitdown_per-1
    st_id[st_id>0.15] = np.nan

    # # 去掉次新股
    stock_info0 = pdb.get_stock_info()
    stock_st0 = pdb.get_stock_st(start, end)
    open_df_per = de_new(open_df_per, stock_info0)
    open_df_per = de_st(open_df_per, stock_st0)

    return open_df_per,close_df_per,ztyizi_open_per,dtyizi_open_per,tingpai_per,st_id


def get_stock_index_weight(start,end,market_index):
        
    # 获取指数成分股权重
    sqlid = ("SELECT* FROM cn_index_components_weights where `index` = '"  
                + market_index + "'and date BETWEEN '" + start + "'and '"+ end +"'")
    indexstock = pd.read_sql_query(sqlid, pdb.conn)
    weight = indexstock.pivot(index='date',columns='ticker',values='weight')
        
    return weight


def get_industry_table(start,end):
    # 读取行业分类数据
    industry_classification = pdb.get_industry_table(start, end)
    industry_classification = industry_preprocess(industry_classification)
    industry_classification = industry_classification.unstack()
    industry_classification.columns = industry_classification.columns.droplevel(0)

    ind = industry_classification.iloc[-1]
    ind_num = ind.drop_duplicates(keep='first',inplace=False).dropna().values
    return industry_classification,ind_num

#  风险敞口配置，按照中证500配比
def get_stock_weight(industry_classification,df_500_weight,stock_id):
    
    df_ind_stock = pd.DataFrame(industry_classification[stock_id])
    count_ind = df_ind_stock.groupby(df_ind_stock.columns[0]).size()
    df_500_weight['industry_num'] = count_ind

    df_500_weight['stock_weight'] = df_500_weight['weights']/df_500_weight['industry_num']
    df_stock_ind = df_ind_stock.reset_index().set_index(df_ind_stock.columns[0])
    df_set = pd.merge(df_500_weight,df_stock_ind,how='outer',left_index=True,right_index=True).dropna()
    stock_id = df_set['ticker']
    df_set['stock_weight'] = df_set['stock_weight']/df_set['stock_weight'].sum()
    #stock_weight = stock_weight/stock_weight.sum()
    df_set = df_set.set_index(df_set['ticker'])
    stock_id = df_set.index
    stock_weight = df_set['stock_weight']
        
    return stock_id,stock_weight   

# 计算风险敞口下的个股的权重
def stock_weight_select(risk_exposure,industry_classification,weights,stock_id):
        
    # zz500的行业权重
    if risk_exposure == 'SH000905':
        df_500_weight = get_industry_weight(industry_classification,weights,100)
        #  风险敞口配置，按照中证500配比
        stock_id,stock_weight = get_stock_weight(industry_classification,df_500_weight,stock_id)
    # 等权重
    elif risk_exposure == None:
        df_stockid = pd.DataFrame(stock_id)
        df_stockid['stock_weight']= 1/len(stock_id)
        stock_weight = df_stockid.set_index('ticker')
        stock_weight =stock_weight['stock_weight']
        
    return stock_id,stock_weight
    

def get_industry_weight(ind,w,numofstock):
    # 计算指数成分股权重占比
    df = pd.merge(ind,w,how='outer',left_index=True,right_index=True)
    df.columns = ['industry_code','weights']
    df_500 = df.dropna()
    df_500_copy = df_500.copy()
    df_500 = df_500.groupby('industry_code').sum()
    df_500['weights'] = df_500['weights']/df['weights'].sum()
    df_500['500_count'] = df_500_copy.groupby('industry_code').count()
    df['weights'].fillna(0,inplace=True)
    df_500['total_count'] = df.groupby('industry_code').count()
    df_500['num_of_stock'] = round(df_500['weights']*numofstock,0)
    df_500['num_of_stock'] = np.where(df_500['num_of_stock']==0,1,df_500['num_of_stock'])
    
    return df_500


def get_stockid_industry(df_500,fa1,ind):
    # 从股票池里选取相对应的指数成分股的个数
    df1 = pd.merge(ind,fa1,how='outer',left_index=True,right_index=True).dropna()
    df1.columns = ['industry_code','factor']
    df1 = df1.reset_index()
    df1 = df1.set_index('industry_code')
    stock_id = pd.DataFrame()

    for i in range(0,len(df_500['num_of_stock'])):

        numofstock = int(df_500['num_of_stock'].iloc[i])
        stock_ind_id = df1.loc[df_500['num_of_stock'].index[i]].sort_values(by=['factor'],ascending = False).iloc[0:numofstock]['ticker']
        #print(stock_id)
        if stock_id.empty:
            stock_id = stock_ind_id
        else:
            stock_id = stock_id.append(stock_ind_id)
    
    return pd.DataFrame(stock_id).set_index('ticker').index

# # 读取回测信息 old
# def get_backtest_fac_info(num_id,start,end):
#     sqlid = "SELECT Columns,Pattern,Adjust_Freq,Pool,Num_Bottom, Weight_Method FROM backtest_to_ainvest where id = "+ str(num_id)
#     ID_SET = pd.read_sql_query(sqlid, pdb.conn)
#     #合成因子数据库
#     factor_name = ID_SET['Columns'][0]
#     pdb_combine = PanelDB()
#     pdb_combine.db_connect('composition_factor')
#     allfac_name = pdb_combine.get_all_factor_list()
#     #print(factor_name)
#     if factor_name in allfac_name :
#         print('get factor from "composition_factor"')
#         factor0 = pdb_combine.get_factor(factor_name, start, end)
#     elif factor_name == 'profit_score':
#         print('get from zyyx')
#         s = "SELECT DISTINCT date,ticker, profit_score FROM `cn_certainty_score_stk` where date BETWEEN '"+ start +"' and '"+ end+ "'"
#         zyyx_date = pd.read_sql_query(s,pdb.conn)
#         factor0 = zyyx_date.pivot(index='date',columns='ticker',values=factor_name)
#     else: 
#         print('get factor from orginal database')
#         factor0 = pdb.get_factor(factor_name, start, end)
#     # 判断是否进行数据清洗
#     facDf = clear_factor(factor0)
        
#     return facDf,ID_SET

# 读取回测信息 new
 
def get_backtest_fac_info(id_set,start,end):

    Pool = id_set['Pool']
    num_of_stock = id_set['Num_Bottom']
    fre = id_set['Adjust_Freq']
    factorname = id_set['Columns']
    num_id = id_set['id']        
    # set barra neu
    size = id_set['size']
    beta = id_set['beta'] 
    style = id_set['style']
    industry = id_set['industry']
    zxh = id_set['neutralize']
    # 风险暴露敞口 ZZ500 或者 A
    risk_exposure = id_set['Weight_Method'] 

    database = id_set['Database']
    pdb_database = PanelDB()
    pdb_database.db_connect(database)
    factorname = id_set['Columns']
    table = id_set['Sheet']
    patt = id_set['Pattern']
    s = "SELECT DISTINCT date,ticker, " + factorname +" FROM " + table + " where date BETWEEN '"+ start +"' and '"+ end+ "'"
    zyyx_date = pd.read_sql_query(s,pdb_database.conn)
    factor0 = zyyx_date.pivot_table(index='date',columns='ticker',values=factorname)
    facDf = clear_factor(factor0)

    return facDf,Pool,num_of_stock,fre,factorname,num_id,size,beta,style,industry,zxh,risk_exposure,patt


# 存入SQL
def save_in_sql(df,name):
    engine = create_engine('mysql+pymysql://wutong_test:qwer1234@rm-bp1vu6g1pj6p9r14cxo.mysql.rds.aliyuncs.com:3306/backtest_result_br')
    df.reset_index().to_sql(name, engine, if_exists = 'append',index=False)


def clear_factor(factor):
    
    if factor.mean().mean() <0.5:
        facDF = factor
    else:      
        fac = dataclear('factor',factor) # 因子清洗
        facDF = fac.inf_clean().log_large().mad_filter(mapping= 'log' , n=4, mapping_pct = 0.5).standardize(method='zscore')
    
    return facDF   

def drop_na(df):
    # 数据处理
    df_index = df[df==1].dropna().index
    return df_index

#%% 回测选股调仓
def stock_pool(ztyizi_open,dtyizi_open,tingpai,st_id,facDFset,t):
    # 剔除一字跌停的个股 停牌的股票 被st的股票   
    zt_open_index_drop = drop_na(ztyizi_open.iloc[t+1])
    dt_open_index_drop = drop_na(dtyizi_open.iloc[t+1])
    tingpai_index_drop = drop_na(tingpai.iloc[t+1])
    st_index_drop = st_id.iloc[t+1].dropna().index
    sid = (facDFset.index.difference(zt_open_index_drop).difference(dt_open_index_drop)
           .difference(tingpai_index_drop).difference(st_index_drop)) 
    return sid,zt_open_index_drop,dt_open_index_drop,tingpai_index_drop,st_index_drop

# 根据中证500权重选择个股  如 有3只农林牧渔的票在500中，那么就选择该板块的得分前三的股票
# 按照申万行业一级分类
def sel_by_ind(method,zf,num_of_stock,facDFset,sid,t,industry_classification,weights):

    if method == 'All A':
        stock_id = facDFset[sid].sort_values(ascending = zf).iloc[0:num_of_stock].index  
        
    elif  method == 'SH000905':
        
        ind = industry_classification.iloc[t] 
        w = weights.iloc[t]
        df_500 = get_industry_weight(ind,w,num_of_stock)
        # 根据行业划分在选股 取每个行业因子值高的前n个 ，效果一般
        stock_id = get_stockid_industry(df_500,facDFset[sid],ind)

    return stock_id

def get_position_open_close(stock_id,total_cap,openset,closeset,stock_weight,TC_buy_pct):
    # 买入股票的开盘和收盘的市值仓位
    openprice = openset[stock_id] 
    closeprice = closeset[stock_id]          
    pos_per_stock = stock_weight*total_cap

    num_of_shares = np.floor(pos_per_stock/(openprice*(1+TC_buy_pct))/100)*100

    TPopen = TP(openset,stock_id,num_of_shares)
    TPClose = TP(closeset,stock_id,num_of_shares)

    return num_of_shares,TPopen,TPClose

# 股票调仓
def stock_classification(stockid_lastperid,sid,dt,tp):
    # 退市票
    stockid_lastperid1 = stockid_lastperid.intersection(sid)
    stockid_tuishi = stockid_lastperid.difference(stockid_lastperid1)
    # 跌停票
    stock_dt = dt.intersection(stockid_lastperid1)
    # 停牌票
    stock_dingpai = tp.intersection(stockid_lastperid1)

    return stockid_lastperid1,stockid_tuishi,stock_dt,stock_dingpai

# 持仓市值
def TP(price,stock_id,num_of_shares_lastperid):
            
    TP_mv = sum(price[stock_id]*num_of_shares_lastperid[stock_id])

    return TP_mv   

#%% 净值计算
def stat(nv):
    # 总收益率
    total_ret_long = nv.iloc[-1]
    # 年化收益率
    an_ret_long = (total_ret_long**(1/(len(nv)/250))-1)
    # 最大回撤
    dd = nv/nv.expanding().max() 
    max_dd =(1 - dd).max() 
    # 夏普
    daily_ret = nv/nv.shift(1) - 1
    an_vol = daily_ret.std()*math.sqrt(252)
    sharp = (an_ret_long - 0.04)/an_vol
    # 卡玛比率
    calmar = (an_ret_long - 0.04)/max_dd
    
    return total_ret_long,an_ret_long,max_dd,sharp,calmar

# 画图
def plot_nv(nv,factorname,stockPool,num_of_stock,fre,num_id):

    nv1 = nv.set_index('As_Of_Date')
    nvplt = (nv1['TAClose']/np.int(nv1['TAClose'][0]))
    nvplt.plot()
    plt.title( 'facname = '+ factorname+ ' ,Pool = '+ stockPool + ' , NumOfstcok = '+ str(num_of_stock)+ ' , fre = '+ str(fre) + ' , ID = '+ str(num_id) )
    

def plot_hedge(market_index,start,end,nv , factorname,stockPool,num_of_stock,fre,num_id,\
               zxh,size,beta,style,industry,savepath,pat):
    
    holding = 1
    nv_date = nv.set_index('As_Of_Date')
    nvplt = nv_date['TAClose']/nv_date['TAClose'].iloc[0]
    df_nv = pd.DataFrame(nvplt)
    
    sqlindex = "SELECT date,close FROM cn_index_daily_price where ticker = '"  + market_index + "'and date BETWEEN '" + start + "'and '"+ end +"'"
    
    indexclose = pd.read_sql_query(sqlindex, pdb.conn)
    indexclose = indexclose.rename(columns={'close':'TAClose','date':'As_Of_Date'})
    indexclose['As_Of_Date'] = pd.to_datetime(indexclose['As_Of_Date'])# 转换日期格式
    indexclose = indexclose.set_index('As_Of_Date')

    nv_index = indexclose / indexclose.iloc[0]

    yld_index = indexclose / indexclose.shift(holding) - 1
    yld_nv = df_nv/df_nv.shift(holding) - 1

    PCT_hedge = yld_nv - yld_index
    NV_hedge = (PCT_hedge+1).cumprod().dropna()


    plt.plot(NV_hedge,label='hedge')
    plt.plot(nv_index,label='index')
    plt.plot(nvplt,label= factorname)
    plt.title( 'facname = '+ factorname \
                           + ' ,Pool = '+ stockPool + ' , NumOfstcok =  ' \
                           + str(num_of_stock)+ ' , fre = ' \
                           + str(fre) + ' , ID = ' \
                           + str(num_id) + ' ,' + zxh ) 
    plt.legend() 
    plt.savefig(savepath+ '/' + factorname + str(fre)+'day' + str(num_of_stock)+'nofs' + '_ID = '+ str(num_id)  + '_zxh=' + zxh + '_size=' + str(size) + '_beta=' + str(beta)+  '_style=' + str(style) + '_ind=' + str(industry) +'_pat='+pat +  '.png', bbox_inches='tight'  ) 
    #plt.savefig('D:\新建文件夹\我的坚果云\单因子检验 封装版 20201231'+ '/' + factorname + str(fre)+'day' + str(num_of_stock)+'nofs' + '_ID = '+ str(num_id)  + '_zxh=' + zxh + '_size=' + str(size) + '_beta=' + str(beta)+  '_style=' + str(style) + '_ind=' + str(industry)  +  '.png', bbox_inches='tight'  ) 
    
    #plt.savefig(sys.path[0] + '/' + factorname + str(fre)+'day' + '_ID = '+ str(num_id)  + '_zxh=' + zxh + '_size=' + str(size) + '_beta=' + str(beta)+  '_style=' + str(style) + '_ind=' + str(industry)  +  '.png', bbox_inches='tight'  ) 

    return NV_hedge,yld_nv ,yld_index

#%%  barra方法优化个股权重
def optimize_weight(hold,exposure,weight):
    
    ex_w = exposure.loc[exposure.ticker.isin(set(weight.ticker))]
    weight = weight.loc[weight.ticker.isin(set(ex_w.ticker))]
    ex_w = ex_w.sort_values(by='ticker').drop('ticker',axis=1)

    ex_w = np.array(ex_w)
    weight = weight.sort_values(by='ticker')
    weight = np.array(weight['weight'])
    benchmark = weight.T.dot(ex_w)

    ex_h = exposure.loc[exposure.ticker.isin(set(hold))].sort_values(by='ticker').drop('ticker',axis=1)
    ex_h = np.array(ex_h)    
    ceil = 2./len(hold)
    fun = lambda w: (w.T.dot(ex_h)-benchmark).dot((w.T.dot(ex_h)-benchmark).T)

    def c1(w):
        # 最低个股百分比
        w = (w>(0.5/len(w))).astype(int)
        idt = np.array([1]*w.shape[0])
        return idt.dot(w)-w.shape[0]
    
    cons = (
        {'type':'eq',
         'fun':lambda w:np.sum(w)-1,
         'jac':lambda x: np.array([1 for _ in range(len(x))])},
        {'type':'ineq',
         'fun':c1}
        )

    limits = tuple((0,0.05) for _ in range(len(hold)))
    w0 = np.array([1./len(hold) for _ in range(len(hold))]).T

    res = minimize(fun,w0,method='SLSQP',constraints=cons,bounds=limits,options={'maxiter':400})

    return res['success'],res['x'],res,fun(w0)

def get_weight(stock_id,date,df_trade_date,df_exposure,df_weights):
    
    prev_date = df_trade_date.loc[date]['ldate']
    tmp_exposure = df_exposure.loc[prev_date].fillna(0)
    tmp_weight = df_weights.loc[prev_date].dropna()
    result,w,_,s = optimize_weight(stock_id,tmp_exposure,tmp_weight)
    w = [x/w.sum() for x in w]
    tmp = pd.Series(w,index=sorted(stock_id))
    return tmp

def barra_data(start,end):
    engine_rq = create_engine('mysql+pymysql://wutong_test:qwer1234@rm-bp1vu6g1pj6p9r14cxo.mysql.rds.aliyuncs.com:3306/original_database')

    s_trade_date = 'select trade_date as date, ltrade_date as ldate from cn_qt_trade_date where exchange=\'001002\''
    df_trade_date = pd.read_sql(sql=s_trade_date,con=engine_rq)
    df_trade_date['date'] = pd.to_datetime(df_trade_date['date'])
    df_trade_date['ldate'] = pd.to_datetime(df_trade_date['ldate'])
    df_trade_date.set_index('date',inplace=True)

    start_1 = df_trade_date.loc[start]['ldate'].iloc[0].strftime('%Y-%m-%d')
    s_weights = 'select * from cn_index_components_weights where date between \'{}\' and \'{}\''.format(start_1,end)
    df_weights = pd.read_sql(sql=s_weights,con=engine_rq)
    df_weights = df_weights.loc[df_weights['index']=='000905.XSHG'][['date','ticker','weight']]
    df_weights['date'] = pd.to_datetime(df_weights['date'])

    s_exposure = 'select * from cn_dy1d_exposure where date between \'{}\' and \'{}\''.format(start_1,end)
    df_exposure0 = pd.read_sql(sql=s_exposure,con=engine_rq)
    df_exposure0['date'] = pd.to_datetime(df_exposure0['date'])

    df_weights.set_index('date',inplace=True)
    df_exposure0.set_index('date',inplace=True)
    
    return df_weights,df_exposure0,df_trade_date

def get_barra_fac(size,beta,style,industry):
    columns = ['ticker']
    if size == 1:
        columns += ['SIZE','SIZENL']
    if beta == 1:
        columns += ['BETA']
    if style == 1:
        columns += ['MOMENTUM','EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
    if industry == 1:
        columns += ['Bank', 'RealEstate', 'Health', 'Transportation', 'Mining',
                    'NonFerMetal', 'HouseApp', 'LeiService', 'MachiEquip', 'BuildDeco',
                    'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics',
                    'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM',
                    'Media', 'IronSteel', 'NonBankFinan', 'ELECEQP', 'AERODEF',
                    'Conglomerates']

    return columns

#%% main bt

@clock 
def trade(total_cap,first_time_openpos,TC_buy_pct,
            TC_sell_pct,facDF,Pool,num_of_stock,fre,
            select_method,factorname,num_id,size,beta,style,
            industry,zxh,risk_exposure,open_df, close_df,
            ztyizi_open, dtyizi_open, tingpai,st_id,zf,
            industry_classification,weights,df_trade_date,df_exposure,df_weights):
    """
    模块
    股票池：剔除st 剔除涨跌停一字板，股票池是HS300 还是zz500
    风险敞口：
    买入规则 仓位分配

    输出股票持仓 买入卖出数量 和相关仓位资金
    t日检测因子
    t+1日的openprice买入
    """

    nv= pd.DataFrame(columns=['id','As_Of_Date','TAOpen','TAClose','TPOpen','TPClose','Cash','TCALL'])
    
    for t in range(0,len(facDF)-1):            
        date = facDF.index[t]            
        if len(facDF.iloc[t].dropna())< 200:
            nv.loc[len(nv)] = [num_id,
                                facDF.iloc[t+1].name,
                                total_cap,
                                total_cap,
                                0,
                                0,
                                total_cap,
                                0]      
            continue    
        # fre天调仓                       
        elif ((t-first_time_openpos) %fre == 0 and t!=first_time_openpos) or (first_time_openpos == -1 and len(facDF.iloc[t].dropna())> 200): 
            print(date)
            # 因子池剔除退市股票
            openset = open_df.iloc[t+1].dropna()
            closeset= close_df.iloc[t+1].dropna()
            facDFset = facDF.iloc[t][openset.index] 

            # 本期股票池剔除一字涨停 一字跌停的个股 停牌的股票 st个股
            sid,zt,dt,tp,st = stock_pool(ztyizi_open,dtyizi_open,tingpai,st_id,facDFset,t)     
            
            if nv.empty:
                print('第一次建仓')
                # 买入股票id
                stock_id = sel_by_ind(select_method,zf,num_of_stock,facDFset,sid,t,industry_classification,weights)
                num_of_shares_lastperid = []
                stock_dt_tp = []
                stock_sell = []
                stock_buy = stock_id
                TPopen_hold=TPopen_sell=TC_sell =TPclose_dingpai = TPclose_dt = TPopen_dt =Cash_left = TPopen_dingpai  = 0       
                first_time_openpos = t
                total_asset_lastperiod = total_cap                
                df_hold = pd.DataFrame(stock_id)
                df_hold.rename(columns={'ticker':facDF.iloc[t].name},inplace= True) 
            
            else:
                print('调仓')
                #上一期股票，退市，跌停和停牌个股    
                stockid_lastperid1,stockid_tuishi,stock_dt,stock_dingpai = stock_classification(stockid_lastperid,sid,dt,tp) 
                
                if stockid_tuishi.empty:
                    tuishi_cap = 0
                else:
                    tuishi_cap = 0
                    for st in range(0,len(stockid_tuishi)):                
                        tuishi_st_stock = open_df[:openset.name][stockid_tuishi[st]].dropna()[-1]*num_of_shares_lastperid[stockid_tuishi[st]]
                        tuishi_cap = tuishi_cap + tuishi_st_stock
                #print(tuishi_cap)
                total_asset_lastperiod = TP(openset,stockid_lastperid1,num_of_shares_lastperid) + tuishi_cap
                
                # 跌停股票市值open close
                TPopen_dt = TP(openset,stock_dt,num_of_shares_lastperid)
                TPclose_dt = TP(closeset,stock_dt,num_of_shares_lastperid)

                # 停牌市值open close
                TPopen_dingpai = TP(openset,stock_dingpai,num_of_shares_lastperid)
                TPclose_dingpai = TP(closeset,stock_dingpai,num_of_shares_lastperid)

                # 本期股票池
                stock_dt_tp = stock_dt.union(stock_dingpai)  
                # 买入股票id
                stock_id = sel_by_ind(select_method,zf,num_of_stock-len(stock_dt_tp),facDFset,sid,t,industry_classification,weights) 


                # hold市值为，用于计算本期本金        
                stock_hold = stock_id.intersection(stockid_lastperid1)
                TPopen_hold = TP(openset,stock_hold,num_of_shares_lastperid)   
                
                # 需要卖出股票的市值，用于计算本期本金
                # 上一期该卖掉的股票的集合。这期股票池和上一期股票池的差集
                stock_sell = stockid_lastperid1.difference(stock_id)
                TPopen_sell = TP(openset,stock_sell,num_of_shares_lastperid) 
                
                # 换手率判断是否调仓
                #Turnover = len(stock_sell)/len(stock_id)
                #boundary = 0.5

                #if Turnover<boundary:
                #    stock_id = stockid_lastperid1
                #    continue

                #需要买入的股票                
                stock_buy = stock_id.difference(stockid_lastperid1) 
                # 手续费
                TC_sell = TP(openset,stock_sell,num_of_shares_lastperid)*TC_sell_pct
                
                # 剩下的本金买入本期剔除停牌和跌停的股票
                total_cap = total_asset_lastperiod - TC_sell - TPopen_dt - TPopen_dingpai + Cash_left 
                
            print('----------')
            print(len(stock_id))
            # barra 中性化
            if zxh == 'barra':
                stock_weight =  get_weight(stock_id,date,df_trade_date,df_exposure,df_weights)
            else:
                stock_id,stock_weight = stock_weight_select(risk_exposure,industry_classification.iloc[t],weights.iloc[t],stock_id)    
            # 选择行业敞口
            #stock_id,stock_weight = stock_weight_select(risk_exposure,industry_classification.iloc[t],weights.iloc[t],stock_id)                
            num_of_shares_ex_dt,TPopen_buy,TPclose_buy = get_position_open_close(stock_id,total_cap,openset,closeset,stock_weight,TC_buy_pct)                                             
            #跌停股票数量+买入股票数量+停牌股票
            num_of_shares = num_of_shares_ex_dt if nv.empty else num_of_shares_lastperid[stock_dt_tp].append(num_of_shares_ex_dt) 
            print(len(stock_dt_tp))
            #所持有股票的集合    
            stock_id= stock_id.union(stock_dt_tp)
            print('行业分类后剩余' + str(len(stock_id)))     
            # 剔除没有行业分类的股票
            stock_buy = stock_buy.intersection(stock_id)
            #手续费
            TC_buy = TP(openset,stock_buy,num_of_shares_ex_dt)*TC_buy_pct    
            tc_total = TC_sell+TC_buy 
            # 持仓市值，手续费
            TPopen =  TPopen_dt + TPopen_dingpai + TPopen_buy      
            TPClose = TPclose_dt + TPclose_dingpai + TPclose_buy 
            Cash_left = total_asset_lastperiod + Cash_left -  tc_total - TPopen        
            TAOpen  = TPopen  + Cash_left 
            TAClose = TPClose + Cash_left

            # 每期股票池集合
            df_shares = pd.DataFrame(num_of_shares,columns=[openset.name]) if nv.empty else pd.concat((df_shares,pd.DataFrame(num_of_shares,columns=[openset.name])),axis=1)  
            #df_shares = pd.concat((df_shares,pd.DataFrame(num_of_shares,columns=[openset.name])),axis=1)  
            # 储存股票集合

            df_hold1 = pd.DataFrame(stock_id)
            df_hold1.rename(columns={'ticker':facDF.iloc[t].name},inplace= True)       
            df_hold = pd.concat((df_hold,df_hold1),axis=1)

        else: # 一直持有

            tc_total = 0 
            openset = open_df.iloc[t+1].dropna()    
            closeset= close_df.iloc[t+1].dropna()
            # 剔除退市股
            stockid_lastperid = stockid_lastperid.intersection(openset.index)        
            TPopen =  TP(openset,stockid_lastperid,num_of_shares_lastperid)
            TPClose = TP(closeset,stockid_lastperid,num_of_shares_lastperid)
            TAOpen  = TPopen  + Cash_left 
            TAClose = TPClose + Cash_left 
            
        # total asset open
        stockid_lastperid = deepcopy(stock_id)
        num_of_shares_lastperid = deepcopy(num_of_shares)
        nv.loc[len(nv)] = [num_id,
                            open_df.iloc[t+1][stock_id].name,
                            TAOpen,
                            TAClose,
                            TPopen,
                            TPClose,
                            Cash_left,
                            tc_total ]
        df_shares = df_shares.fillna(0)
    
    return nv, df_shares

def trading_stock_list(df_shares,num_id):
        df_shares_trades = df_shares - df_shares.shift(axis=1,periods = 1)
        df_shares_trades[df_shares_trades.columns[0]] = df_shares[df_shares_trades.columns[0]]
        shares_stack = df_shares_trades.T.stack()
        df =  pd.DataFrame(shares_stack)
        df=df.reset_index()
        df['id'] = num_id
        df = df.set_index('id')
        df.rename(columns={'level_0':'AS_Of_Date','level_1':'Ticker',0:'Shares'},inplace=True)
        
        return df