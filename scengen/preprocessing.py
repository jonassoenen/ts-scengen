from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class YearlyInfoPreprocessor:
    def __init__(self, columns_to_use, normalized=True):
        self.scaler = MinMaxScaler() if normalized else None
        self.columns_to_use = columns_to_use

    def fit(self, info_df):
        info = info_df.loc[:, self.columns_to_use]
        if self.scaler is not None:
            self.scaler.fit(info)

    def transform(self, info_df):
        info_df = info_df[self.columns_to_use]
        if self.scaler is not None:
            return pd.DataFrame(
                self.scaler.transform(info_df),
                index=info_df.index,
                columns=info_df.columns,
            )
        return info_df

    def fit_transform(self, info_df):
        self.fit(info_df)
        return self.transform(info_df)


def yearly_profile_df_to_daily_df(df):
    """
    Converts from a yearly dataframe (i.e. every row in the dataframe is a year of data) to a daily dataframe
    with a multi-index where each row is a day
    """
    all_dates = pd.date_range("2016-01-01", "2016-12-31")
    all_profiles = df.index
    columns = pd.date_range("2016-01-01 0:00", periods=96, freq="15min")
    data = df.to_numpy().reshape((-1, 96))
    return pd.DataFrame(
        data,
        index=pd.MultiIndex.from_product(
            [all_profiles, all_dates], names=["meterID", "date"]
        ),
        columns=columns,
    )
