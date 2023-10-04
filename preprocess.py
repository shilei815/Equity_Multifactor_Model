import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from common import *


def gen_descriptive_head():
    """
    生成描述性统计的空DataFrame
    """

    return pd.DataFrame(columns = ['name','count','mean','min','1%','2%','3%','4%','5%','10%','15%','20%','25%','50%','75%','80%','85%','90%','95%','96%','97%','98%','99%','max','skew','kurtosis','var','std'])

def descriptive_stats(name: str, data: pd.DataFrame):
    """
    对矩阵式面板数据全体数据做描述性统计
    """

    values = data.unstack().dropna().tolist()

    return {'name': name,
        'count': len(values),
        'mean': np.mean(values), 
        'min': np.min(values),
        '1%': np.percentile(values, 1),
        '2%': np.percentile(values, 2),
        '3%': np.percentile(values, 3),
        '4%': np.percentile(values, 4),
        '5%': np.percentile(values, 5),
        '10%': np.percentile(values, 10),
        '15%': np.percentile(values, 15),
        '20%': np.percentile(values, 20),
        '25%': np.percentile(values, 25),
        '50%': np.median(values),
        '75%': np.percentile(values, 75),
        '80%': np.percentile(values, 80),
        '85%': np.percentile(values, 85),
        '90%': np.percentile(values, 90),
        '95%': np.percentile(values, 95),
        '96%': np.percentile(values, 96),
        '97%': np.percentile(values, 97),
        '98%': np.percentile(values, 98),
        '99%': np.percentile(values, 99),
        'max': np.max(values),
        'skew': stats.skew(values),
        'kurtosis': stats.kurtosis(values),
        'var': np.var(values),
        'std': np.std(values)}

def inf_clean(data: pd.DataFrame, fillna: str = "ffill"):
    """
    对数据中的inf值（+-99999999）改变为nan，默认用ffill法处理掉nan值
    """

    data[data == 99999999] = np.nan
    data[data == -99999999] = np.nan
    data = data.fillna(method = fillna)
    return data


def log_large(data: pd.DataFrame):
    """
    定义取log值的标准
    ** 有市值中性化的含义在里面
    """
    
    if (abs(data).median(axis = 1)).median() > 500:
        data = np.log(data)
    return data


def get_data_align_marks(*args):
    """
    输入应该等长等宽的DataFrame，输出使其等长等宽的行号index和列号columns
    """

    index = set(args[0].index)
    columns = set(args[0].columns)
    for i in range(1, len(args)):
        index = index.intersection(set(args[i].index))
        columns = columns.intersection(set(args[i].columns))
    return (sorted(index), sorted(columns))

def de_outlier(data: pd.DataFrame, method: str = 'mad', mapping: str = 'log', **kwargs):
    """
    去极值函数：所有参数必须带参数名称！！！
    method和mapping参数可以缺省：分别代表用mad发现极值、极值处理方法为clip

    MAD法：n为上下n个MAD，映射区间设置为再往上下n*mapping_pct的区间内
    如：de_outlier(method = "mad", mapping = None, n = 60, mapping_pct = 0.25)
    3-sigma法：n为上下n个sigma，映射区间设置为再往上下mapping_n的区间内
    如：de_outlier(method = "3sigma", mapping = None, n = 3, mapping_n = 1)

    百分位法：下界为min_pct%，上界为max_pct%，m映射区间设置为再往上下mapping_pct%的区间内
    如：de_outlier(method = "pct", mapping = None, min_pct = 1.5, max_pct = 98.5, mapping_pct = 0.5)
    """
    
    if method == 'mad':
        return mad_filter(data, mapping, kwargs.get('n', 60), kwargs.get('mapping_pct', 0.25))
    if method == '3sigma':
        return three_sigma_filter(data, mapping, kwargs.get('n', 3.0), kwargs.get('mapping_n', 1.0))
    if method == 'pct':
        return percentile_filter(data, mapping, kwargs.get('min_pct', 1.5), kwargs.get('max_pct', 98.5), kwargs.get('mapping_pct', 0.5))

def mad_filter(data: pd.DataFrame, mapping: str, n: float, mapping_pct: float):
    """
    Parameter:
        factor_name: name of factors in Wind. (str)
        n: how many times new median. (int)
    Return:
        filtered data. (pd.DataFrame)
    """

    median = data.median(axis = 1)
    mad = (abs((data.T - median).T)).median(axis = 1)

    min_range = median - n * mad
    max_range = median + n * mad
    if mapping is None:
        return data.clip(min_range, max_range, axis=0)
    elif mapping is not None:
        min_thr = min_range - n * mapping_pct * mad
        max_thr = max_range + n * mapping_pct * mad
        return data_mapping(data, min_range, max_range, min_thr, max_thr, mapping)

def three_sigma_filter(data: pd.DataFrame, mapping: str, n: float, mapping_n: float):
    """
    Parameter:
        factor_name: name of factors in Wind. (str)
        n: how many sigmas. (int)
    Return:
        filtered data. (pd.DataFrame)
    """

    mean = data.mean(axis = 1)
    sigma = data.std(axis = 1)

    min_range = mean - n * sigma
    max_range = mean + n * sigma
    if mapping is None:
        return data.clip(min_range, max_range, axis=0)
    elif mapping is not None:
        min_thr = min_range - mapping_n * sigma
        max_thr = max_range + mapping_n * sigma
        return data_mapping(data, min_range, max_range, min_thr, max_thr, mapping)

def percentile_filter(data: pd.DataFrame, mapping: str, min_pct: float, max_pct: float, mapping_pct: float):
    """
    Parameters:
        factor_name: name of factors in Wind. (str)
        min_pct: minimum percentage. (float)
        max_pct: maximum percentage. (float)
    Return:
        filtered data. (pd.DataFrame)
    """

    min_range = pd.Series([np.percentile(data.iloc[i], min_pct) for i in range(len(data))], index = data.index)
    max_range = pd.Series([np.percentile(data.iloc[i], max_pct) for i in range(len(data))], index = data.index)
    if mapping is None:
        return data.clip(min_range, max_range, axis=0)
    elif mapping is not None:
        min_thr = pd.Series([np.percentile(data.iloc[i], min_pct - mapping_pct) for i in range(len(data))], index = data.index)
        max_thr = pd.Series([np.percentile(data.iloc[i], max_pct + mapping_pct) for i in range(len(data))], index = data.index)
        return data_mapping(data, min_range, max_range, min_thr, max_thr, mapping)

def data_mapping(data: pd.DataFrame, min_range: float, max_range: float, min_thr: float, max_thr: float, mapping: str):
    """
    根据mapping参数，选择去极值的映射方法
    """

    if mapping == 'linear':
        return data_mapping_linear(data, min_range, max_range, min_thr, max_thr)
    if mapping == "log":
        return data_mapping_log(data, min_range, max_range, min_thr, max_thr)

def data_mapping_linear(data: pd.DataFrame, min_range: float, max_range: float, min_thr: float, max_thr: float):
    """
    线性映射法
    """

    data_T = data.T
    data_T[data_T > max_range] = max_range + (max_thr - max_range) / (data_T.max() - max_range) * (data_T[data_T > max_range] - max_range)
    data_T[data_T < min_range] = min_range - (min_range - min_thr) / (min_range - data_T.min()) * (min_range - data_T[data_T < min_range])

    return data_T.T

def data_mapping_log(data: pd.DataFrame, min_range: float, max_range: float, min_thr: float, max_thr: float):
    """
    log函数映射法
    """

    data_T = data.T
    data_T[data_T > max_range] = max_range + np.log(1 + data_T[data_T > max_range] - max_range)
    data_T[data_T < min_range] = min_range - np.log(1 + min_range - data_T[data_T < min_range])

    return data_T.T

def standardize(data: pd.DataFrame, method: str = "zscore"):
    """
    标准化接口
    """

    if method == "zscore":
        return zscore(data)

def zscore(data: pd.DataFrame):
    """
    Parameter:
        factor_name: name of factors in Wind. (str)
        start_year:the start_year the data start
    Return:
        standardized and Filtered (MAD) data. (pd.DataFrame)
    """

    mean = data.mean(axis = 1)
    std = data.std(axis = 1)
    return ((data.T - mean) / std).T

def de_new(data: pd.DataFrame, stock_info: pd.DataFrame):
    """
    去掉次新股（182天）
    """

    if type(stock_info['listed_date'][0]) == str:
        stock_info['listed_date'] = stock_info.apply(lambda row: datetime.strptime(row['listed_date'], '%Y-%m-%d'), axis = 1)
    for i in range(len(stock_info)):
        stock_name = stock_info['order_book_id'][i]
        listed_date = stock_info['listed_date'][i]
        limit_date = listed_date + timedelta(days = 182)
        if limit_date.year < 2999:
            data.loc[data.index < limit_date, stock_name] = np.nan
    return data

def de_st(data: pd.DataFrame, stock_st: pd.DataFrame):
    """
    去掉ST股
    """

    data = data.drop(list(set(data.columns.tolist())-set(stock_st.columns.tolist())), axis = 1)
    stock_st = stock_st.drop(list(set(stock_st.columns.tolist())-set(data.columns.tolist())), axis = 1)
    data = data[stock_st.applymap(lambda element: not element)]
    return data

def industry_preprocess(industry_table: pd.DataFrame, method: str = 'delete'):
    """
    识别并处理一股票同时分属多行业的情况
    """

    if method == 'delete':
        return industry_preprocess_delete(industry_table)

def industry_preprocess_delete(industry_table: pd.DataFrame):
    """
    识别并删除一股票同时分属多行业的情况
    """

    industry_table = industry_table.reset_index()
    industry_table = industry_table.drop_duplicates(['date','ticker'], keep = False)
    industry_table = industry_table.set_index(['date','ticker'])
    return industry_table


def neutralize(data: pd.DataFrame, market_value: pd.DataFrame, industry_table: pd.DataFrame, method: str = 'stepwise_mean', **kwargs):
    """
    中性化接口
    """

    if method == 'stepwise_mean':
        return neutralize_stepwise_mean(data, market_value, industry_table)


def neutralize_stepwise_mean(data: pd.DataFrame, market_value: pd.DataFrame, industry_table: pd.DataFrame):
    """
    两种顺序的两步骤中性化取平均值
    """

    data1 = neutralize_stepwise(data, market_value, industry_table, first = 'market_value')
    data2 = neutralize_stepwise(data, market_value, industry_table, first = 'industry')
    data = (data1 + data2) / 2
    return data


def neutralize_stepwise(data: pd.DataFrame, market_value: pd.DataFrame, industry_table: pd.DataFrame, first: str):
    """
    两步骤中性化，first决定先跑什么
    """

    if first == 'market_value':
        data = neutralize_market_value(data, market_value)
        data = neutralize_industry(data, industry_table)
    elif first == 'industry':
        data = neutralize_industry(data, industry_table)
        data = neutralize_market_value(data, market_value)
    return data


def neutralize_market_value(data: pd.DataFrame, market_value: pd.DataFrame):
    """
    市值中性化
    """

    new_data = pd.DataFrame(columns = data.columns)
    for i in range(len(data)):
        data_cs = data.iloc[i]
        market_value_cs = market_value.iloc[i]
        resid = regress(market_value_cs, data_cs).resid
        resid.name = data_cs.name
        new_data = new_data.append(resid)
    return new_data

def neutralize_industry(data: pd.DataFrame, industry_table: pd.DataFrame):
    """
    行业中性化
    """

    new_data = pd.DataFrame(columns = data.columns)
    for i in range(len(data)):
        data_cs = data.iloc[i]

        # industry_matrix = industry_table[industry_table['date'] == data_cs.name].drop('date', axis = 1)

        date_set = industry_table.index
        if data_cs.name in date_set:
            industry_matrix = industry_table.loc[data_cs.name]
            industry_matrix['values'] = 1
            industry_matrix = industry_matrix.reset_index()
            industry_matrix = industry_matrix.set_index(['ticker','industry_code'])
            industry_matrix = industry_matrix.unstack()
            industry_matrix.columns = industry_matrix.columns.droplevel(level = 0)
            industry_matrix = industry_matrix.fillna(0)
            industry_matrix_pre = industry_matrix
        else:
            industry_matrix = industry_matrix_pre

        data_cs = data_cs.drop(list(set(data_cs.index.tolist())-set(industry_matrix.index.tolist())))
        industry_matrix = industry_matrix.drop(list(set(industry_matrix.index.tolist())-set(data_cs.index.tolist())))

        resid = regress(industry_matrix, data_cs).resid
        resid.name = data_cs.name
        new_data = new_data.append(resid)
    return new_data
