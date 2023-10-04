import pandas as pd

from dbbean import *

class PanelDB():
    """
    面板数据与数据库的交互类
    """
    
    history = None
    db = None

    def __init__(self):
        """
        初始化数据库信息
        """

        self.db = DBBean()

    def table_to_panel(self, data_table: pd.DataFrame):
        """
        将三列式长表数据转换为矩阵式面板数据
        """

        data = data_table.unstack()
        data.columns = data.columns.droplevel(0)
        return data

    def get_history(self, start: str, end: str):
        """
        获得所有行情信息
        """

        self.db.init(default='wutong_rds_mysql')
        self.db.db_connect('original_database')
        sql = "select * from cn_stock_daily_price where date >= '" + start + "' and date <= '" + end + "'"
        self.history = self.db.exec_query(sql).set_index(['date','ticker'])
        sql = "select date,ticker,total from cn_floating_stock where date >= '" + start + "' and date <= '" + end + "'"
        total = self.db.exec_query(sql).set_index(['date','ticker'])
        self.history = self.history.merge(total, on = ['date','ticker'])
        self.history['market_value'] = self.history['open'] * self.history['total']

    def is_four_price_doji(self, s: pd.Series):
        """
        判断某一行是否是一字线（用于lambda表达式）
        """

        if s['open'] != s['close']:
            return False
        elif s['open'] != s['high']:
            return False
        elif s['open'] != s['low']:
            return False
        else:
            return True

    def de_four_price_doji(self):
        """
        去掉所有一字线数据
        """

        self.history = self.history.iloc[self.history.apply(lambda s: not self.is_four_price_doji(s), axis = 1).tolist()]
        
    def de_small_size(self, size: float):
        """
        去掉指定小市值的股票
        """
        
        self.history = self.history[self.history['market_value'] >= size]

    def get_price(self, name: str):
        """
        从已获得的行情数据获取对应数据的宽表
        """

        price_table = pd.DataFrame(self.history[name])
        price = self.table_to_panel(price_table)
        return price
    
    def get_factor_table(self, factor_name: str, table_name: str, start: str, end: str, frequency: str, source: str, if_local: int):
        """
        获取三列式长表型因子数据
        """
        if if_local == 1:
            if frequency == 'daily':
                sql = "select date,ticker," + factor_name + " from " + table_name + " where date >= '" + start + "' and date <= '" + end + "'"
            else:
                sql = "select date,ticker,value as " + factor_name + " from " + table_name + " where date >= '" + start + "' and date <= '" + end + "' and factor_name='" + factor_name + "'"
        else:
            self.db.init(default='gpg_rds_sqlserver')
            if source == 'TLSJ':
                self.db.db_connect('TLSJ')
                start = pd.to_datetime(start).strftime('%Y%m%d')
                end = pd.to_datetime(end).strftime('%Y%m%d')
                sql = "select trade_date as date,ticker_symbol as ticker,%s from %s where trade_date>='%s' and trade_date<='%s'" % (factor_name, table_name, start, end)
            else:
                sql = ""

        factor_table = self.db.exec_query(sql)
        if if_local==0 and source=='TLSJ':
            factor_table['date'] = factor_table['date'].apply(lambda x: str(x))
            factor_table['date'] = pd.to_datetime(factor_table['date'])
            factor_table['ticker'] = factor_table['ticker'].apply(lambda x: ticker_change(x))
        factor_table = factor_table.set_index(['date','ticker'])
        return factor_table.sort_index()

    def get_factor(self, factor_name: str, table_name: str, start: str, end: str, type: str = 'original'):
        """
        获取矩阵式面板数据
        """
        self.db.init(default='wutong_rds_mysql')
        if type == 'original':
            self.db.db_connect('original_database')
        elif type == 'composition':
            self.db.db_connect('composition_factor')

        sql = "select frequency,source,if_local from cn_factor_info where factor_name='%s' and factor_table='%s'" % (factor_name, table_name)
        result_of_query = self.db.exec_query(sql)
        frequency = result_of_query.iloc[0,0]
        source = result_of_query.iloc[0,1]
        if_local = result_of_query.iloc[0,2]

        factor_table = self.get_factor_table(factor_name, table_name, start, end, frequency, source, if_local)
        factor = self.table_to_panel(factor_table)
        return factor.fillna(method = 'ffill')


    def get_factor_name_list(self, table_name: str, type: str = 'original'):
        """
        根据表名获得因子的名称
        """

        self.db.init(default='wutong_rds_mysql')
        if type == 'original':
            self.db.db_connect('original_database')
        elif type == 'composition':
            self.db.db_connect('composition_factor')

        sql = "select factor_name from cn_factor_info where factor_table='%s'" % (table_name)
        factor_name_list = self.db.exec_query(sql).iloc[:,0].tolist()
        return factor_name_list

    def get_factor_table_list(self, type: str = 'original'):
        """
        根据当前表中所有因子表的名称
        """

        self.db.init(default='wutong_rds_mysql')
        if type == 'original':
            self.db.db_connect('original_database')
        elif type == 'composition':
            self.db.db_connect('composition_factor')

        sql = "select distinct factor_table from cn_factor_info"
        factor_table_list = self.db.exec_query(sql).iloc[:,0].tolist()
        return factor_table_list


    def get_industry_table(self, start: str, end: str):
        """
        获取股票的行业信息表
        """

        sql = "select As_Of_Date as date,Ticker as ticker,Industry_Code as industry_code from industry_classification where As_Of_Date >= '" + start + "' and As_Of_Date <= '" + end + "'"
        industry_table = self.db.exec_query(sql)
        industry_table['ticker'] = industry_table['ticker'].apply(lambda x: ticker_change(x))
        return industry_table.set_index(['date','ticker'])

    def get_stock_info(self):
        """
        获取股票信息
        """

        sql = "select * from cn_stock_info"
        stock_info = self.db.exec_query(sql)
        return stock_info

    def get_stock_st(self, start: str, end: str):
        """
        获取ST股票信息
        """

        sql = "select * from cn_stock_st where date >= '" + start + "' and date <= '" + end + "'"
        stock_st = self.db.exec_query(sql).set_index(['date','ticker'])
        stock_st['is_st'] = stock_st.apply(lambda r: (r == 1))
        stock_st = self.table_to_panel(stock_st)
        return stock_st

    def get_cross_section_ticker_factor_from_table(self, date: str, table_name: str):
        """"""
        
        self.db.init(default = 'wutong_rds_mysql')
        self.db.b_connect('original_database')
        sql = "select factor_name where factor_table='%s'" % (table_name)
        factor_name_list = self.db.exec_query(sql)
        if len(factor_name_list) == 0:
            self.db.init(default = 'wutong_rds_mysql')
            self.db.db_connect('composition_factor')
            factor_name_list = self.db.exec_query(sql)
            if len(factor_name_list) == 0:
                return None

        factor_name_list = factor_name_list['factor_name'].tolist()
        sql = "select ticker," + ",".join(factor_name_list) + " from " + table_name + " where date='" + date + "'"
        df = self.db.exec_query(sql)
        return df

    def get_index_weights(self, start: str, end: str, index: str = '000905.XSHG'):
        """
        获取index中个股权重
        """
        
        sql = "select * from cn_index_components_weights where date >= '" + start + "' and date <= '" + end + "'"
        index_weights = self.db.exec_query(sql)
        index_weights = index_weights.loc[index_weights['index'] == index][['date','ticker','weight']]
        index_weights = self.table_to_panel(index_weights.set_index(['date','ticker']))
        return index_weights