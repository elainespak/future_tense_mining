import pandas as pd


class Portfolio():
    """Portfolio class holds portfolio data and details.

    Args:
        score_path (str): path of csv file containing gvkey, score, and filing date
        num_bin (int): number of bins to use when constructing long-short portfolio

    Attributes:
        score (dataframe): gvkey, gramm_score, filed, quantile info, rebalance_dates
        stocks (list): stocks universe
        rebalance_dates (list): rebalance dates
        portfolio (dataframe): position of stock universe at a rebalance date

    Raises:
        AssertionError: when columns are not in the dataframe
    """
    def __init__(self, score_path, num_bin):
        self.num_bin = num_bin
        self.score = pd.read_csv(score_path, encoding='utf-8')[['gvkey', 'gramm_score', 'filed']]
        assert 'gvkey' in self.score.columns
        self.stocks = list(set(self.score.gvkey))
        self.score = self.construct_rebalance_dates(self.score)
        self.portfolio = self.construct_portfolio(self.score)

    def construct_rebalance_dates(self, df):
        """Appends 'rebalance_date' to df."""

        assert 'filed' in df.columns
        df['filed'] = pd.to_datetime(df['filed'])

        suffix_list = ['-02-01 00:00:00','-05-01 00:00:00','-08-01 00:00:00','-11-01 00:00:00']
#         suffix_list = ['-01-20 00:00:00', '-04-20 00:00:00','-07-20 00:00:00','-10-20 00:00:00']
#         suffix_list = ['-02-10 00:00:00','-05-10 00:00:00','-08-10 00:00:00','-11-10 00:00:00']
        dates = list(set([x for x in df['filed']]))
        years = list(set([date.year for date in dates]))
        years.sort()
        rebalance_dates = [pd.Timestamp(str(years[0]))] + \
                        [pd.Timestamp(str(year)+suffix) for year in years for suffix in suffix_list] + \
                            [pd.Timestamp(str(years[-1]+1)+suffix_list[0])]
        rebalance_dates = [date for date in rebalance_dates if (date >= min(dates)) & (date <= max(dates))]

        df['rebalance_date'] = pd.cut(df['filed'], rebalance_dates, right=False, labels = rebalance_dates[1:])
        df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
        df = df.dropna()
        self.rebalance_dates = list(set(df['rebalance_date']))

        return df

    def construct_portfolio(self, df):
        """Filters baskets of highest/lowest scores, and construct long-short portfolios."""
        df = df.sort_values('filed')
        df = df.drop_duplicates(['rebalance_date', 'gvkey'], keep='last')
        df['score_quantile'] = df.groupby('rebalance_date')['gramm_score'].apply(lambda x: pd.qcut(x, self.num_bin, labels=range(1,self.num_bin+1)))
        df['score_quantile'] = df['score_quantile'].astype(int)
        portfolio = df.pivot_table(values='score_quantile', columns='gvkey', index='rebalance_date')
        portfolio.index.name = 'date'
        portfolio.index = pd.to_datetime(portfolio.index)
        portfolio = portfolio.fillna(0).sort_index()

        return portfolio
