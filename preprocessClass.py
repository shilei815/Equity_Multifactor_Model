import pandas as pd
import numpy as np

from datetime import datetime, timedelta


class dataclear():
    """
    数据清洗
    
    """
    
    def __init__(self,
                facname,
                DF
                ):
        """
        初始化数据库信息
        """
        self.facname = facname
        self.DF = DF
    
    def inf_clean(self):
        """
        对数据中的inf值（+-99999999）改变为nan，默认用ffill法处理掉nan值
        """
        self.DF[self.DF == 99999999] = np.nan
        self.DF[self.DF == -99999999] = np.nan
        return self

    
    def log_large(self):
        """
        定义取log值的标准
        ** 有市值中性化的含义在里面
        """
    
        if (abs(self.DF).median(axis = 1)).median() > 500:
            self.DF = np.log(self.DF)
        return self
    
    
    def standardize(self, method: str = "zscore"):
        """
        标准化接口
        """

        if method == "zscore":
            return self.zscore()

    def zscore(self):
        """
        Parameter:
        factor_name: name of factors in Wind. (str)
        start_year:the start_year the data start
        Return:
        standardized and Filtered (MAD) data. (pd.DataFrame)
        """

        mean = self.DF.mean(axis = 1)
        std = self.DF.std(axis = 1)
        
        return ((self.DF.T - mean) / std).T
    
    
#     def de_outlier(self, method: str = 'mad', mapping: str = 'log', **kwargs):
    
#         if method == 'mad':
#             return self.mad_filter(self, mapping, kwargs.get('n', 60), kwargs.get('mapping_pct', 0.25))
    
    
    def mad_filter(self, mapping: str, n: float, mapping_pct: float):
        """
        Parameter:
            factor_name: name of factors in Wind. (str)
            n: how many times new median. (int)
        Return:
            filtered data. (pd.DataFrame)
        """
        median = self.DF.median(axis = 1)
        mad = (abs((self.DF.T - median).T)).median(axis = 1)

        min_range = median - n * mad
        max_range = median + n * mad
        if mapping is None:
            return self.DF.clip(min_range, max_range, axis=0)
        elif mapping is not None:
            min_thr = min_range - n * mapping_pct * mad
            max_thr = max_range + n * mapping_pct * mad
            return self.data_mapping(min_range, max_range, min_thr, max_thr, mapping)
        
        
    def data_mapping(self, min_range: float, max_range: float, min_thr: float, max_thr: float, mapping: str):
        """
        根据mapping参数，选择去极值的映射方法
        """

        if mapping == 'linear':
            return self.data_mapping_linear( min_range, max_range, min_thr, max_thr)
        if mapping == "log":
            return self.data_mapping_log( min_range, max_range, min_thr, max_thr)
    
    def data_mapping_linear(self, min_range: float, max_range: float, min_thr: float, max_thr: float):
        """
        线性映射法
        """
        data_T = self.DF.T
        data_T[data_T > max_range] = max_range + (max_thr - max_range) / (data_T.max() - max_range) * (data_T[data_T > max_range] - max_range)
        data_T[data_T < min_range] = min_range - (min_range - min_thr) / (min_range - data_T.min()) * (min_range - data_T[data_T < min_range])

        return self

    def data_mapping_log(self, min_range: float, max_range: float, min_thr: float, max_thr: float):
        """
        log函数映射法
        """
        data_T = self.DF.T
        data_T[data_T > max_range] = max_range + np.log(1 + data_T[data_T > max_range] - max_range)
        data_T[data_T < min_range] = min_range - np.log(1 + min_range - data_T[data_T < min_range])

        return self



