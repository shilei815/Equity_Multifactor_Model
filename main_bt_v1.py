# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 23:04:30 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
from paneldb import *
import time
from copy import deepcopy
from preprocessClass import *
from preprocess import*
from backtest_func import *
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from copy import deepcopy
import csv
from pandas import read_csv
import schedule
import time
# import sys
# sys.path
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False 

# get stock data
start = '2018-7-1'
end = '2020-12-31'

pdb = PanelDB()
pdb.db_connect('original_database')
# get stock data
stcok_data = get_stock_data(start, end)
open_df_per,close_df_per,ztyizi_open_per,dtyizi_open_per,tingpai_per,st_id = stcok_data

# 获取行业权重 指数成分股占比
market_index =  '000905.XSHG'
weights = get_stock_index_weight(start,end,market_index)
# 指数的行业权重
industry_classification , ind_num= get_industry_table(start,end)
# get barra data
df_weights,df_exposure0,df_trade_date = barra_data(start,end)

#%% 循环调用回测函数backtest 
def sleeptime(hour,min,sec):
    return hour*3600 + min*60 + sec;

# 调时间回测
second = sleeptime(0,0,30);
id_list = pd.DataFrame()

id_start = 621
id_end = 624
save_tradingdata = 1

while 1==1:
    time.sleep(second);
    print('do action');
    ID_SET = pdb.get_id_set_br()
    print(len(ID_SET))
    localtime = time.asctime( time.localtime(time.time()) )
    print("本地时间为 :", localtime)
    
    if id_list.empty:
        id_list=ID_SET
        print('id_list初始化')
    else:
        #id_series = set(id_list['id']).difference(set(ID_SET['id']))
        id_series = list(set(id_list['id'])^(set(ID_SET['id'])))
        
        if not id_series and id_start==0:
            print('没有新的回测id')
            continue 
        elif id_series or id_start!=0:
            
            id_list=ID_SET
            
            if id_series:
                # id_start = id_series[0]
                # id_end = id_series[-1]
                index_start = ID_SET[ID_SET['id']==id_series[0]].index[0]
                index_end =   ID_SET[ID_SET['id']==id_series[-1]].index[0]
            else: 
                # id_start = id_series[0]
                # id_end = id_series[-1]
                # print('id_start=' + str(id_start)+ '\nid_end=' + str(id_end))
                
                # bt ID的区间
                #id_start = 218
                #id_end = 219
                index_start = ID_SET[ID_SET['id']==id_start].index[0]
                index_end =   ID_SET[ID_SET['id']==id_end].index[0]
                
                
            df_stat = pd.DataFrame(columns=['ID','total_ret','annual_ret','max_dd','sharp','calmar'])
                
            # 读取回测信息
            for i in range(index_start,index_end+1):
                
                id_set = ID_SET.iloc[i]        
                """
                初始设置
                """
                total_cap = 10000000000
                # 买入佣金 万3
                TC_buy_pct = 3*0.01*0.01
                # 卖出佣金 万13
                TC_sell_pct = 3*0.01*0.01+0.001
                
                select_method = 'All A'
                
                print(id_set)
                """
                回测信息和参数
                """
                # get factor因子值
                facDF,Pool,num_of_stock,fre,factorname,num_id,size,beta,style,industry,zxh,risk_exposure,patt = get_backtest_fac_info(id_set,start,end)    
                print('get factor')
                columns = get_barra_fac(int(size),beta,style,industry)
                df_exposure = df_exposure0.copy()
                df_exposure = df_exposure[columns]
                # 对齐
                facDF.index = pd.to_datetime(facDF.index)
                index, columns = get_data_align_marks(open_df_per, close_df_per, facDF)
                open_df  = open_df_per.loc[index, columns]
                close_df = close_df_per.loc[index, columns]
                facDF = facDF.loc[index, columns] # 与yld对齐
                ztyizi_open = ztyizi_open_per.loc[index, columns]
                dtyizi_open = dtyizi_open_per.loc[index, columns]
                tingpai= tingpai_per.loc[index, columns]
            
                """
                模式识别
                """
                if patt == 'linear+':
                    zf = False
                elif patt == 'linear-':
                    zf = True
                elif patt == 'convex':
                    facDF = abs(facDF.sub(facDF.median(axis=1),axis=0))
                    print('convex')
                    zf = True        
                
                
                # t+1日的open价格为买入价格
                backtest_trades = pd.DataFrame(columns=['Date','Ticker','Shares'])
                print('get all data')
             
                # 调用回测函数
                first_time_openpos= -1
                # 输出净值和交易明细
                nv, df_shares = trade(total_cap,first_time_openpos,TC_buy_pct,TC_sell_pct,
                                      facDF,Pool,num_of_stock,fre,select_method,
                                      factorname,num_id,
                                      size,beta,style,industry,zxh,risk_exposure,
                                      open_df, close_df, ztyizi_open, dtyizi_open, tingpai,st_id,zf,
                                      industry_classification,weights,df_trade_date,df_exposure,df_weights)
                
                nv1 = nv.set_index('id')
                # # 交易明显
                df_stocklist = trading_stock_list(df_shares,num_id)
                
                # # 净值数据存入数据库数据库
                name = 'backtest_summary'
                save_in_sql(nv1,name) if save_tradingdata == 1 else print('do not save')
                # 交易明显存入数据库数据库
                name = 'backtest_trades'
                save_in_sql(df_stocklist,name) if save_tradingdata == 1 else print('do not save')
                
            
                #画图W
                plot_nv(nv1,factorname,Pool,num_of_stock,fre,num_id)
                fig = plt.figure()
                # 工作站
                savepath = "C:/Users/Administrator/Desktop/因子回测结果"
                # borun
                #savepath = "C:/Users/borun\Desktop/新建文件夹 (2)"
                NV_hedge,yld_nv ,yld_index = plot_hedge(market_index,start,end,nv1,factorname,Pool,num_of_stock,fre,num_id,zxh,size,beta,style,industry,savepath,id_set['Pattern'])
                total_ret_long,an_ret_long,max_dd,sharp,calmar = stat(NV_hedge)
                df_stat.loc[len(df_stat)] = [ num_id,total_ret_long[0],an_ret_long[0],max_dd[0],sharp[0],calmar[0]]
                
            
                filename = '近3年 backtest_summary_stats_br.csv'
                data = read_csv('近3年 backtest_summary_stats_br.csv')
                #data = data.set_index('ID')
                data.loc[len(data)] = [num_id,factorname,total_ret_long[0],an_ret_long[0],max_dd[0],sharp[0],calmar[0],zxh,size,beta,style,industry,fre,num_of_stock,Pool,id_set['Pattern']]
                data.set_index('ID').to_csv(filename)

        id_start = 0
