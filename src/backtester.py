from dateutil.relativedelta import relativedelta
import pandas as pd
import

class Backtester():
    def __init__(self, portfolio, price_path, option):
        assert option in ['ls', 'quantile']
        self.portfolio = portfolio.portfolio
        self.option = option
        self.num_bin = portfolio.num_bin
        self.portfolio = self.portfolio.append(pd.Series(name=self.portfolio.index[-1]+relativedelta(months=3)))
        self.stocks = portfolio.stocks
        self.rebalance_dates = portfolio.rebalance_dates
        self.gvkey_to_tic = None

        self.price = pd.read_csv(price_path, encoding='utf-8')
        self.price = self.pre_process_price(self.price)
        self.portfolio_returns = self.backtest(self.price, self.portfolio)

    def pre_process_price(self, price):
        assert {'gvkey', 'prccd', 'ajexdi', 'datadate'} <= set(price.columns)
        price = price[price.gvkey.isin(self.stocks)].copy()
        price['iid'] = price['iid'].astype(str)
        price = price.sort_values('iid')
        stocks = price[['gvkey', 'iid']].drop_duplicates(['gvkey'], keep='first')
        price = stocks.merge(price, how='left', on=['gvkey','iid'])
        if 'isin' in price.columns:
            price.loc[:, 'tic'] = price['isin'].str[3:9]
        elif 'tic' not in price.columns:
            raise AssertionError
        self.gvkey_to_tic = price[['gvkey', 'tic']].set_index('gvkey').to_dict()
        price.loc[:, 'datadate'] = pd.to_datetime(price['datadate'], format='%Y%m%d')
        price.loc[:, 'adj_closing_price'] = price['prccd'] / price['ajexdi']
        price = price[['gvkey', 'tic', 'datadate', 'adj_closing_price']].copy()
        price = price.sort_values('datadate')
        price = price.pivot_table('adj_closing_price', ['datadate'], 'gvkey')
        price = price[price.isnull().sum(axis=1)/len(price.columns) < 0.8]

        return price

    def backtest(self, price, portfolio):
        i = 0
        returns = []
        portfolio_returns_list = [pd.Series([])]*self.num_bin
        for date, row in portfolio.iterrows():
            if i < len(portfolio)-1:
                price_list = []
                if self.option == 'ls':
                    stocks_to_invest = row[row.isin([1, self.num_bin])].replace({self.num_bin:-1})
                    price_list = [price[stocks_to_invest.index]]
                else:
                    for i in range(self.num_bin):
                        price_list += [price[row[row==i+1].index]]
                print(len(price_list))
                for idx, price_quarter in enumerate(price_list):
                    print(idx)
                    price_quarter = price_quarter[(price_quarter.index <= date+relativedelta(months=3)) & (price_quarter.index >= date)]
                    price_quarter = price_quarter.sort_index()
                    returns_quarter = (price_quarter.shift(-1) / price_quarter) -1
                    returns_quarter = returns_quarter.iloc[:-1]
                    if self.option == 'ls':
                        portfolio_returns_list[idx] = portfolio_returns_list[idx].append(pd.Series(returns_quarter.mul(stocks_to_invest).mean(axis=1), index=returns_quarter.index))
                    else:
                        portfolio_returns_list[idx] = portfolio_returns_list[idx].append(pd.Series(returns_quarter.mean(axis=1), index=returns_quarter.index))
            i += 1

        return portfolio_returns_list
