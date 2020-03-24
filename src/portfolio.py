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
        self.score = pd.read_csv(score_path, encoding='utf-8')[['gvkey', 'gramm_score', 'filed']]
        assert 'gvkey' in self.score.columns
        self.stocks = list(set(self.score.gvkey))
        self.score = self.construct_rebalance_dates(self.score)
        self.portfolio = self.construct_portfolio(self.score, num_bin)

    def construct_rebalance_dates(self, df):
        """Appends 'rebalance_date' to df."""

        assert 'filed' in df.columns
        df['filed'] = pd.to_datetime(df['filed'])
        suffix_list = ['-01-20 00:00:00','-04-20 00:00:00','-07-20 00:00:00','-10-20 00:00:00']
        dates = list(set([x for x in df['filed']]))
        years = list(set([date.year for date in dates]))
        years.sort()
        rebalance_dates = [pd.Timestamp(str(years[0]))] + \
                        [pd.Timestamp(str(year)+suffix) for year in years for suffix in suffix_list] + \
                            [pd.Timestamp(str(years[-1]+1)+suffix_list[0])]
        rebalance_dates = [date for date in rebalance_dates if (date >= min(dates)) & (date <= max(dates))]

        df['rebalance_date'] = pd.cut(df['filed'], rebalance_dates, labels = rebalance_dates[1:])
        df['rebalance_date'] = pd.to_datetime(df['rebalance_date'])
        df = df.dropna()
        self.rebalance_dates = list(set(df['rebalance_date']))
        return df

    def construct_portfolio(self, df, num_bin):
        """Filters baskets of highest/lowest scores, and construct long-short portfolios."""

        df['score_quantile'] = df.groupby('rebalance_date', level=0)\
                                .apply(pd.DataFrame.sort_values, 'gramm_score', ascending=False)['gramm_score']\
                                .transform(lambda x: pd.qcut(x, num_bin, labels=range(1,num_bin+1)))
        df = df[df['score_quantile'].isin([1, num_bin])]
        portfolio = pd.DataFrame(columns=['date'] + self.stocks)
        portfolio['date'] = self.rebalance_dates
        portfolio = portfolio.set_index('date')
        for idx, row in df.iterrows():
            if row.rebalance_date in portfolio.index:
                if row.score_quantile == 1:
                    portfolio.loc[pd.to_datetime(row.rebalance_date), row.gvkey] = 1 # bug in pandas - datetime is converted to int when looping
                elif row.score_quantile == num_bin:
                    portfolio.loc[pd.to_datetime(row.rebalance_date), row.gvkey] = -1
        portfolio = portfolio.fillna(0)

        return portfolio
