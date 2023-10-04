import pandas as pd
import statsmodels.api as sm

def regress(x : pd.DataFrame, y: pd.Series, method: str = 'OLS'):
    """
    一个序列或矩阵与另一个序列做回归，不同回归方法的接口
    """

    if method == 'OLS':
        return regress_OLS(x, y)

def regress_OLS(x : pd.DataFrame, y: pd.Series):
    """
    两个序列（时间或截面）做回归，用OLS法
    """

    X = sm.add_constant(x, has_constant = 'add')
    model = sm.OLS(y, X, missing = 'drop')
    return model.fit()

def arima(y: pd.Series, p: int, d: int, q: int):
    """
    
    """