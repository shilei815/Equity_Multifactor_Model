import pandas as pd
import pymysql
from sqlalchemy import create_engine



class PanelDB():
    """
    面板数据与数据库的交互类
    """
    
    host = ""
    port = ""
    username = ""
    password = ""
    schema = ""
    db = None
    conn = None
    cursor = None
    history = None

    def __init__(self,
                host: str = "rm-bp1vu6g1pj6p9r14cxo.mysql.rds.aliyuncs.com",
                port: str = "3306",
                username: str = "wutong_test",
                password: str = "qwer1234"
                ):
        """
        初始化数据库信息
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.schema = ""

    def db_connect(self, schema: str):
        """
        打开数据库连接
        """

        self.schema = schema
        self.db = pymysql.connect(host=self.host, user = self.username,password= self.password, database = self.schema)
        #self.db = pymysql.connect(self.host, self.username, self.password, self.schema)
        self.conn = create_engine('mysql+pymysql://' + self.username + ':' + self.password + '@' + self.host + ':' + self.port + '/' + self.schema)
        self.cursor = self.db.cursor()

    def db_close(self):
        """
        关闭数据库
        """

        self.db.close()
        self.conn.dispose()

    def exec_query(self, sql: str):
        """"""

        self.db_connect(self.schema)
        return pd.read_sql_query(sql, self.conn)

    def table_to_panel(self, data_table: pd.DataFrame):
        """
        将三列式数据转换为矩阵式面板数据
        """

        data = data_table.unstack()
        data.columns = data.columns.droplevel(0)
        return data

    def get_factor_table(self, name: str, table_name: str, start: str, end: str, data_frequency: str):
        """
        获取三列式因子数据
        """
        if data_frequency == 'daily':
            sql = "select DISTINCT date,ticker," + name + " from " + table_name + " where date >= '" + start + "' and date <= '" + end + "'"
        else:
            sql = "select DISTINCT date,ticker,value as " + name + " from " + table_name + " where date >= '" + start + "' and date <= '" + end + "' and factor_name='" + name + "'"

        factor_table = self.exec_query(sql).set_index(['date','ticker'])
        return factor_table.sort_index()

    def get_factor(self, name: str, start: str, end: str):
        """
        获取矩阵式面板数据
        """

        sql = "select factor_table,frequency from cn_factor_info where factor_name='" + name + "'"
        table_name = self.exec_query(sql).iloc[0,0]
        data_frequency = self.exec_query(sql).iloc[0,1]

        factor_table = self.get_factor_table(name, table_name, start, end, data_frequency)
        factor = self.table_to_panel(factor_table)
        return factor.fillna(method = 'ffill')

    def get_history(self, start: str, end: str):
        """
        获得所有行情信息
        """

        sql = "select * from cn_stock_daily_price where date >= '" + start + "' and date <= '" + end + "'"
        self.history = self.exec_query(sql).set_index(['date','ticker'])
        sql = "select DISTINCT date,ticker,total from cn_stock_floating_shares where date >= '" + start + "' and date <= '" + end + "'"
        total = self.exec_query(sql).set_index(['date','ticker'])
        self.history = self.history.merge(total, on = ['date','ticker'])
        self.history['market_value'] = self.history['open'] * self.history['total']
        # self.history = self.history.drop('total', axis = 1)

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
        # return self.history

    def get_price(self, name: str):
        """
        从已获得的行情数据获取对应数据的宽表
        """

        price_table = pd.DataFrame(self.history[name])
        price = self.table_to_panel(price_table)
        return price


    def get_factor_list(self, table_name: str):
        """
        根据表名获得因子的名称
        """

        sql = "select factor_name from cn_factor_info where factor_table='" + table_name + "'"
        factor_list = self.exec_query(sql).iloc[:,0].tolist()
        return factor_list

    def get_all_factor_list(self):
        """
        获取所有已入库的因子名称
        """

        sql = "select factor_name from cn_factor_info"
        factor_list = self.exec_query(sql).iloc[:,0].tolist()
        return factor_list

    def ticker_change(self, ticker: str, raw: str = 'SW'):
        """
        ticker转换器，raw取SW 或 RQ
        """

        if raw == 'SW':
            return self.ticker_change_SW2RQ(ticker)
        elif raw == 'RQ':
            return self.ticker_change_RQ2SW(ticker)

    def ticker_change_SW2RQ(self, ticker: str):
        """
        ticker转换器，申万转米筐
        """

        ex = ticker[0:2]
        code = ticker[2:]
        if ex == 'SH':
            ex = 'XSHG'
        else:
            ex = 'XSHE'
        res = code + '.' + ex
        return res

    def ticker_change_RQ2SW(self, ticker: str):
        """
        ticker转换器，米筐转申万
        """

        strs = ticker.split('.')
        if strs[1] == 'XSHE':
            strs[1] = 'SZ'
        else:
            strs[1] = 'SH'
        res = strs[1]+strs[0]
        return res

    def get_industry_table(self, start: str, end: str):
        """
        获取股票的行业信息表
        """

        sql = "select As_Of_Date as date,Ticker as ticker,Industry_Code as industry_code from industry_classification where As_Of_Date >= '" + start + "' and As_Of_Date <= '" + end + "'"
        industry_table = self.exec_query(sql)
        industry_table['ticker'] = industry_table.apply(lambda row: self.ticker_change(row['ticker']), axis = 1)
        return industry_table.set_index(['date','ticker'])
    
    def get_industry_table2(self, start: str, end: str):
        """
        获取股票的行业信息表
        """

        sql = "select As_Of_Date as date,Ticker as ticker,Industry_Code as industry_code from industry_classification_2nd where As_Of_Date >= '" + start + "' and As_Of_Date <= '" + end + "'"
        industry_table = self.exec_query(sql)
        industry_table['ticker'] = industry_table.apply(lambda row: self.ticker_change(row['ticker']), axis = 1)
        return industry_table.set_index(['date','ticker'])
    
    
    def get_stock_info(self):
        """
        获取股票信息
        """

        sql = "select * from cn_stock_info"
        stock_info = self.exec_query(sql)
        return stock_info

    def get_stock_st(self, start: str, end: str):
        """
        获取ST股票信息
        """

        sql = "select * from cn_stock_st where date >= '" + start + "' and date <= '" + end + "'"
        stock_st = self.exec_query(sql).set_index(['date','ticker'])
        stock_st['is_st'] = stock_st.apply(lambda r: (r == 1))
        stock_st = self.table_to_panel(stock_st)
        return stock_st
    
    def get_id_set_br(self):
        """
        获取回测信息
        """
        
        sqlid = "SELECT * FROM backtest_to_br"
        ID_SET = self.exec_query(sqlid)
        return ID_SET


        
    def getmarketdata(self,market_index : str,start : str,end : str):
        
        sqlindex = "SELECT date,close FROM cn_index_daily_price where ticker = '"  + market_index + "'and date BETWEEN '" + start + "'and '"+ end +"'"
        indexclose = self.exec_query(sqlindex)
        #indexclose = pd.read_sql_query(sqlindex, self.pdb.conn)
        indexclose = indexclose.rename(columns={'close':'TAClose','date':'As_Of_Date'})
        indexclose['As_Of_Date'] = pd.to_datetime(indexclose['As_Of_Date'])# 转换日期格式
        indexclose = indexclose.set_index('As_Of_Date')
        
        return indexclose