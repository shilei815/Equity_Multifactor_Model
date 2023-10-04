import pymysql
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import math
from sqlalchemy import create_engine
from numpy import *
#from statsmodels.tsa.arima_model import ARIMA
from copy import deepcopy
import time
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings 

class fea_Ttest():
    def __init__(self,
                 fea_T: pd.DataFrame,
                 period: int, 
                 Tup:float, 
                 Tdown:float
                 ):
        """
        初始化数据库信息
        """
        self.fea_T = fea_T
        self.period = period
        self.Tup = Tup
        self.Tdown = Tdown
        self.N = ""
        self.ratio = ""

    def stats_sign_ratio(self,):        
        """
        计算T的T>1.96和小于-1.96的比例
        """
        T_ts = []
        for t in range(1,len(self.fea_T),self.period):
            Tvalue_lin = self.fea_T[t:self.period+t].mean()/( (self.fea_T[t:self.period+t]).std()/math.sqrt(self.period-1))
            T_ts.append(Tvalue_lin)
        T_ts[-1]= self.fea_T[-self.period:].mean()/((self.fea_T[-self.period:].std())/math.sqrt(self.period-1))                                       
        #return  T_ts
        return T_ts #self.T_per(T_ts)

    def T_per(self, T_ts):

        T_period_Ts = pd.DataFrame(T_ts).T
        UP_T = pd.DataFrame(np.where((T_period_Ts>self.Tup),1,0))
        down_T = pd.DataFrame(np.where((T_period_Ts<self.Tdown),1,0))
        # 计算大于1.96 小于-1.96的比例
        up_per = UP_T.T.sum()/len(UP_T.iloc[0])
        down_per = down_T.T.sum()/len(down_T.iloc[0])
        return up_per,down_per  

    def pre_Method(self,method:str, N:int, ratio:float):
        # 滚动计算一段样本内的预测结果
        self.N = N
        self.ratio = ratio
        ts =    self.fea_T.iloc[-self.N:]
        train = ts.iloc[0: math.ceil(int(self.ratio*(self.N)))]
        valid = ts.iloc[math.ceil(int(self.ratio*(self.N))):]  

        if method == 'arima':
            return self.arima(ts,train,valid)

    def arima(self,ts,train,valid):
        Tbeta_set = []
        Tbeta_Tvalues_set = []      

        for i in range(0,len(ts.columns)):

            beta1,beta1_T,model = arima_all(train.iloc[:,i],valid.iloc[:,i])
            Tbeta_set.append(beta1)
            Tbeta_Tvalues_set.append(beta1_T)

            beta2,beta2_T = arima_fixpara(train.iloc[:,i],valid.iloc[:,i], model)                         
            Tbeta_set.append(beta2)
            Tbeta_Tvalues_set.append(beta2_T)
    
        return Tbeta_set,Tbeta_Tvalues_set

    def arima_all(self,train,valid):

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
        print("arima整体训练预测值和样本值的beta为","%.2f"%beta1, "beta的T值为",                "%.2f"%beta1_T)
        print("预测样本数量"+  "%.0f"%len(valid))

        return beta1,beta1_T,model
    
    def arima_fixpara(train,valid,model):
    
        history = [x for x in train]
        predictions = list()    
        for t in range(len(valid)):
            output = model.fit_predict(history,n_periods=1)
            predictions.append(output)
            obs = valid[t]
            history.append(obs)
            history = history[1:]

        #plt.savefig( 'Z:\投研\策略\Alpha策略\多因子模型\jpg/'  + fac_name                      +'arima' +'.png', bbox_inches='tight')
        y_MA_stabpara = valid
        x_MA_stabpara = predictions

        res_MA_stabpara = regress(y_MA_stabpara,x_MA_stabpara)
        beta1_T = res_MA_stabpara.tvalues[1]
        beta1 = res_MA_stabpara.params[1]
        #plt.scatter(y_MA_stabpara,x_MA_stabpara)
        print("arima锁住参数滚动法：预测值和样本值的beta为","%.2f"%beta1, "beta的T值                为","%.2f"%beta1_T)

        return beta1,beta1_T