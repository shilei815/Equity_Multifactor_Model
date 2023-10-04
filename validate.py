import pandas as pd
import numpy as np
from common import *
import matplotlib.pyplot as plt
import math
from pmdarima.arima import auto_arima
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def gen_cross_section_df(factor: pd.DataFrame, yld: pd.DataFrame, row_index: int,holding:int):
    """
    结合收益率数据合成当期界面数据集
    """

    x = factor.iloc[row_index]
    y = yld.iloc[row_index +1 + holding]
    return pd.DataFrame({'factor': x, 'yield': y}).dropna()

def gen_cross_section_stats(cs_df: pd.DataFrame):
    """
    结合特定收益率数据做一次截面统计
    """

    reg_result = regress(cs_df['factor'], cs_df['yield'])
    return {'t_alpha': reg_result.tvalues[0],
        't_beta': reg_result.tvalues[1], 
        'alpha': reg_result.params[0],
        'beta': reg_result.params[1],
        'ic': cs_df.corr().iloc[1, 0],
        'spearman': cs_df.corr('spearman').iloc[1, 0]}

def subgroup(cs_df: pd.DataFrame, n_group: int = 10, method: str = 'percentile'):
    """
    对截面数据进行分组，不同方法的接口
    """

    cs_df['subgroup'] = 0
    if method == 'percentile':
        return subgroup_percentile(cs_df, n_group)

def subgroup_percentile(cs_df: pd.DataFrame, n_group: int):
    """
    用因子的百分位数分组
    """

    for i in range(n_group):
        bottom = np.percentile(cs_df['factor'],  i * 100 / n_group)
        cs_df['subgroup'] = np.where(cs_df['factor'] >= bottom, i, cs_df['subgroup'])
    return cs_df

def gen_group_means(grouped_cs_df: pd.DataFrame):
    """
    输入做好分组的界面数据集，返回该截面的因子和收益数据的每组均值
    """

    meaned_cs_df = grouped_cs_df.groupby('subgroup').mean()
    
    return (meaned_cs_df['factor'].values.tolist(), meaned_cs_df['yield'].values.tolist())

def pattern_recognition(grouped_cs_df: pd.DataFrame, method = 'linear'):
    """
    模式识别接口
    """

    if method == 'linear':
        return regress(grouped_cs_df['factor'], grouped_cs_df['yield']).tvalues[1]
    if method == 'extreme_end':
        return pattern_recognition_extreme_end(grouped_cs_df)
    if method == 'convex':
        return pattern_recognition_convex(grouped_cs_df)

def pattern_recognition_extreme_end(grouped_cs_df: pd.DataFrame):
    """
    两端极值
    """

    n_group = 10

    grouped_cs_df['POO'] = np.where(grouped_cs_df['subgroup'] == n_group - 1, 1, 0)
    grouped_cs_df['NOO'] = np.where(grouped_cs_df['subgroup'] == n_group - 1, -1, 0)
    grouped_cs_df['OOP'] = np.where(grouped_cs_df['subgroup'] == 0, 1, 0)
    grouped_cs_df['OON'] = np.where(grouped_cs_df['subgroup'] == 0, -1, 0)

    return (regress(grouped_cs_df['POO'], grouped_cs_df['yield']).tvalues[1],
        regress(grouped_cs_df['NOO'], grouped_cs_df['yield']).tvalues[1],
        regress(grouped_cs_df['OOP'], grouped_cs_df['yield']).tvalues[1],
        regress(grouped_cs_df['OON'], grouped_cs_df['yield']).tvalues[1])

def pattern_recognition_convex(grouped_cs_df: pd.DataFrame):
    """
    突形（U形） 中间突起
    """

    grouped_cs_df['re_factor'] = abs(grouped_cs_df['factor'] - np.mean(grouped_cs_df['factor']))
    grouped_cs_df = grouped_cs_df.drop(columns = ['subgroup'])
    re_group = subgroup(grouped_cs_df)
    return regress(re_group['re_factor'], re_group['yield']).tvalues[1]

def feature_engineer(grouped_cs_df: pd.DataFrame, method: str = 'awt'):
    """
    特征工程接口
    """

    if method == "awt":
        return feature_engineer_awt(grouped_cs_df)

def feature_engineer_awt(grouped_cs_df: pd.DataFrame):
    """
    特征工程 爱玩特方法 分出3类共六种模式进行模式识别
    """

    extreme_end_result = pattern_recognition(grouped_cs_df, method = 'extreme_end')
    return {'linear': pattern_recognition(grouped_cs_df),
        'POO': extreme_end_result[0],
        'NOO': extreme_end_result[1],
        'OOP': extreme_end_result[2],
        'OON': extreme_end_result[3],
        'convex': pattern_recognition(grouped_cs_df, method = 'convex')}

def get_cross_section_panels(factor: pd.DataFrame, yld: pd.DataFrame, holding:int,n_group: int = 10): 
    """
    得到截面在时序上检验结果的面板数据
    """

    cross_section_stats_panel = pd.DataFrame()
    group_factor_mean_panel = pd.DataFrame(columns = [i for i in range(n_group)])
    group_yield_mean_panel = pd.DataFrame(columns = [i for i in range(n_group)])
    feature_engineer_t_panel = pd.DataFrame()
    index_list = []

    for i in range(0,len(factor) - 2 - holding, holding ):

        cs_df = gen_cross_section_df(factor, yld, i, holding)
       
        if len(cs_df) == 0:
            continue

        grouped_cs_df = subgroup(cs_df, n_group)
        
        if len(grouped_cs_df['subgroup'].drop_duplicates()) < 3:
            continue
        
        cross_section_stats_panel = cross_section_stats_panel.append(gen_cross_section_stats(cs_df), ignore_index = True)
        
        group_factor_mean_panel.loc[len(group_factor_mean_panel)], group_yield_mean_panel.loc[len(group_yield_mean_panel)] = gen_group_means(grouped_cs_df)
        
        feature_engineer_t_panel = feature_engineer_t_panel.append(feature_engineer(grouped_cs_df), ignore_index = True)
        index_list.append(factor.index[i])
        
    cross_section_stats_panel.index = index_list
    group_factor_mean_panel.index = index_list
    group_yield_mean_panel.index = index_list
    feature_engineer_t_panel.index = index_list
    
    return (cross_section_stats_panel, group_factor_mean_panel, group_yield_mean_panel, feature_engineer_t_panel)


def statplot(factorname,group_yield_mean_panel,cross_section_stats_panel,holding):
    one_ten = (group_yield_mean_panel.iloc[:,0]-group_yield_mean_panel.iloc[:,-1] +1).cumprod(axis=0)

    plt.figure(figsize=(10, 10))
    plt.subplot(331)
    plt.title(factorname +" 的alpha时间序列")
    plt.plot(cross_section_stats_panel.iloc[:,0].values)
    plt.subplot(332)
    plt.title(factorname +" 的IC时间序列")
    plt.plot(cross_section_stats_panel.iloc[:,2].values)
    plt.subplot(333)
    plt.title(factorname +" 的spearman时间序列")
    plt.plot(cross_section_stats_panel.iloc[:,3].values)
    plt.subplot(334)
    plt.title(factorname +" 的T_Beta时间序列")
    plt.plot(cross_section_stats_panel.iloc[:,5].values)
    plt.subplot(335)
    plt.title(factorname +" 收益排序柱状图")
    plt.bar(group_yield_mean_panel.mean().index, group_yield_mean_panel.mean())
    plt.subplot(336)
    plt.title(factorname +" 第一组 - 第十组" + str(holding)+"天")
    plt.plot(one_ten)
    plt.savefig( 'C:\\Users\\Administrator\\Desktop\\replicates/' + factorname + str(holding)+"天" +'v1'+'.png', bbox_inches='tight')

    NV = (group_yield_mean_panel+1).T.cumprod(axis=1)
    plt.figure(figsize=(5, 5))
    plt.plot(NV.T)  
    plt.title(factorname +'_' + str(holding) + "天_分组收益率曲线")
    plt.savefig( 'C:\\Users\\Administrator\\Desktop\\replicates/'  + factorname + str(holding)+"天" +'v2'+'.png', bbox_inches='tight')
    
    return NV.iloc[0][-1],NV.iloc[-1][-1]

def cal_T(feature_engineer_t_panel,period,Tup,Tdown):  
    T_lin_ts = []
    for t in range(1,len(feature_engineer_t_panel),period):
        Tvalue_lin = feature_engineer_t_panel[t:period+t].mean()/((feature_engineer_t_panel[t:period+t]).std()/math.sqrt(period-1))
        T_lin_ts.append(Tvalue_lin)
    T_lin_ts[-1]= feature_engineer_t_panel[-period:].mean()/((feature_engineer_t_panel[-period:].std())/math.sqrt(period-1))
    return T_per(T_lin_ts, Tup ,Tdown,period)


def T_per(T_lin_ts, Tup, Tdown, period):

    T_period_Ts = pd.DataFrame(T_lin_ts).T

    UP_T = pd.DataFrame(np.where((T_period_Ts>Tup),1,0))
    down_T = pd.DataFrame(np.where((T_period_Ts<Tdown),1,0))
    # 计算大于1.96 小于-1.96的比例
    up_per = UP_T.T.sum()/len(UP_T.iloc[0])
    down_per = down_T.T.sum()/len(down_T.iloc[0])
    return up_per,down_per   

def predict_arima(test_N:int, train_ratio:float, feature_engineer_t_panel : pd.DataFrame):
    Tbeta_set = []
    Tbeta_Tvalues_set = []
    """
    # arima检测
    # 1 train计算整体预测valid
    # 2 train计算参数不变滚动预测valid     
    """
    ts =feature_engineer_t_panel.iloc[-test_N:] # 最后500个样本

    # 滚动计算一段样本内的预测结果
    train = ts.iloc[0: math.ceil(int(train_ratio*(test_N)))]
    valid = ts.iloc[math.ceil(int(train_ratio*(test_N))):]  

    for i in range(0,len(ts.columns)):

        beta1,beta1_T,model = arima_all(train.iloc[:,i],valid.iloc[:,i])
        Tbeta_set.append(beta1)
        Tbeta_Tvalues_set.append(beta1_T)

        Tbeta,Tbeta_Tvalues = arima_fixpara(train.iloc[:,i],valid.iloc[:,i],model)
        Tbeta_set.append(Tbeta)
        Tbeta_Tvalues_set.append(Tbeta_Tvalues)
    
    return Tbeta_set,Tbeta_Tvalues_set

def arima_all(train,valid):

    model = auto_arima(train, start_p=1, start_q=1,
                  information_criterion='bic',
                  test='adf',       # use adftest to find optimal 'd'
                  max_p=5, max_q=5, # maximum p and q
                  m=1,              # frequency of series
                  d=None,           # let model determine 'd'
                  seasonal=False,   # No Seasonality
                  start_P=0, 
                  D=0, 
                  trace=False,
                  error_action='ignore',  
                  suppress_warnings=True, 
                  stepwise=False,
                  max_order = None)

    output = model.predict(len(valid))
    # H0:mean样本 = mean预测值
    # 做回归后beta的t值 > 1.96显著不为0，则拒绝原假设
    y_all = valid
    x_all = output

    res1 = regress(y_all,x_all)
    beta1_T = res1.tvalues[1]
    beta1 = res1.params[1]
    #plt.scatter(y_all,x_all)
    print("预测样本数量"+  "%.0f"%len(valid))
    print("arima整体训练预测值和样本值的beta为","%.2f"%beta1, "beta的T值为","%.2f"%beta1_T)
    

    return beta1,beta1_T,model

# 2 滚动预测ARIMA模型锁住参数
# 固定样本40 120 250 200 50  最大样本
def arima_fixpara(train,valid,model):
    
    history = [x for x in train]
    predictions = list()    
    for t in range(len(valid)):
        output = model.fit_predict(history,n_periods=1)
        predictions.append(output)
        obs = valid[t]
        history.append(obs)
        history = history[1:]

    #plt.savefig( 'Z:\投研\策略\Alpha策略\多因子模型\jpg/'  + fac_name +'arima'+'.png', bbox_inches='tight')
    y_MA_stabpara = valid
    x_MA_stabpara = predictions
    #run_regression(y_MA_stabpara,x_MA_stabpara)

    res_MA_stabpara = regress(y_MA_stabpara,x_MA_stabpara)
    beta1_T = res_MA_stabpara.tvalues[1]
    beta1 = res_MA_stabpara.params[1]
    #plt.scatter(y_MA_stabpara,x_MA_stabpara)
    print("arima锁住参数滚动法：预测值和样本值的beta为","%.2f"%beta1, "beta的T值为","%.2f"%beta1_T)

    return beta1,beta1_T

def test_stat(nv,holding):
    # 总收益率
    total_ret_long = nv.iloc[-1]
    # 年化收益率
    an_ret_long = (total_ret_long**(1/(len(nv)/250*holding))-1)
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