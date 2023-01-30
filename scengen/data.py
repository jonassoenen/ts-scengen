import numpy as np
import pandas as pd

from scengen.preprocessing import yearly_profile_df_to_daily_df


def generate_mockup_data(profiles=None, yearly_attributes=None, daily_attributes=None, seed = None):
    """
        Generates some mock-up data
    """
    # default attribute values
    if profiles is None:
        profiles = [f"profile{i}_2016" for i in range(100)]
    if yearly_attributes is None:
        yearly_attributes = ["yearly_consumption", "connection_power", "y_attr2"]
    if daily_attributes is None:
        daily_attributes = ["feelsLikeC", "isWeekend", 'dayOfWeek', 'month']

    random = np.random.default_rng(seed)

    # generate the yearly electricity consumption time series data
    timestamps = pd.date_range("2016-01-01 0:00", "2016-12-31 23:45", freq="15min")
    yearly_data = random.random((len(profiles), len(timestamps)))
    yearly_data_df = pd.DataFrame(yearly_data, index=profiles, columns=timestamps)

    # transform this yearly data to a daily data format as well
    daily_data_df = yearly_profile_df_to_daily_df(yearly_data_df)

    # generate the yearly info dataframe
    yearly_info = random.random((len(profiles), len(yearly_attributes)))
    yearly_info_df = pd.DataFrame(
        yearly_info, index=profiles, columns=yearly_attributes
    )

    # generate the daily info dataframe
    daily_info = random.random((len(daily_data_df.index), len(daily_attributes)))
    daily_info_df = pd.DataFrame(
        daily_info, index=daily_data_df.index, columns=daily_attributes
    )

    # return the result
    return yearly_data_df, daily_data_df, yearly_info_df, daily_info_df
