import abc

import numpy as np
import pandas as pd

from scengen.rlp import read_rlps, read_slps


class Generator(metaclass=abc.ABCMeta):
    """
    A generator generates new scenarios instead of just sampling historical data.
    """

    @abc.abstractmethod
    def fit(self, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_samples(self, yearly_info_df, daily_info_df, nb_samples=100):
        raise NotImplementedError()


class RLPGenerator(Generator):
    def __init__(self, yearly_data_df):
        # yearly data df is necessary to figure out when to use which SLP!
        # same for yearly info df
        self.low_nightday_ratio = self.yearly_data_df_to_low_nightday_ratio(
            yearly_data_df
        )

        self.slps = None
        self.rlps = None

    def yearly_data_df_to_low_nightday_ratio(self, yearly_data_df):
        years = yearly_data_df.index.str.slice(-5, -1).astype("int")
        is_category_s21 = pd.Series(index=yearly_data_df.index, dtype="bool")

        for year, year_df in yearly_data_df.groupby(years):
            is_leap_year = pd.to_datetime(f"{year}-01-01").is_leap_year
            if not is_leap_year:
                # drop februari 29
                year_df = year_df.drop(
                    columns=pd.date_range(
                        "2016-02-29 00:00", "2016-02-29 23:45", freq="15min"
                    ).to_list()
                )
            # reindex with correct year
            year_df = year_df.set_axis(
                year_df.columns.map(lambda t: t.replace(year=year)), axis=1
            )

            # boolean array that indicates which columns to use for day/night tarif
            is_weekday = year_df.columns.weekday < 5
            is_night = (year_df.columns.hour < 7) | (year_df.columns.hour > 21)
            night_tarif = ~is_weekday | is_night
            day_tarif = ~night_tarif

            # calculate consumption during day/night tarif periods
            day_consumption = year_df.loc[:, day_tarif].sum(axis=1)
            night_consumption = year_df.loc[:, night_tarif].sum(axis=1)

            # calculate the ratio
            night_day_ratio = night_consumption / day_consumption
            is_category_s21.loc[year_df.index] = night_day_ratio < 1.3

        return is_category_s21

    def yearly_info_df_to_slp_category(self, yearly_info_df):
        slp_categories = pd.Series(index=yearly_info_df.index, dtype="object")

        residentials = yearly_info_df.index[
            yearly_info_df.consumer_type == "residential"
        ]
        residential_low_nightday_ratio = self.low_nightday_ratio[residentials]
        s21_profiles = residential_low_nightday_ratio[
            residential_low_nightday_ratio
        ].index
        s22_profiles = residential_low_nightday_ratio[
            ~residential_low_nightday_ratio
        ].index
        slp_categories[s21_profiles] = "S21"
        slp_categories[s22_profiles] = "S22"

        professionals_s11 = (yearly_info_df.consumer_type == "professional") & (
            yearly_info_df.connection_capacity < 56
        )
        professionals_s12 = (yearly_info_df.consumer_type == "professional") & (
            yearly_info_df.connection_capacity >= 56
        )

        slp_categories.loc[professionals_s11] = "S11"
        slp_categories.loc[professionals_s12] = "S12"
        return slp_categories

    def fit(self, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df):
        self.slps = read_slps()
        self.rlps = read_rlps()
        print("loaded data")

    def get_slp_to_use_for_year(self, year, slp: pd.DataFrame, yearly_info_df):

        slp_categories_per_profile = self.yearly_info_df_to_slp_category(yearly_info_df)
        slp = slp.set_axis(["S21", "S22", "S11", "S12"], axis=1)

        slp_profile_per_profile = pd.Series(index=yearly_info_df.index, dtype="object")
        for category, group in slp_profile_per_profile.groupby(
            slp_categories_per_profile
        ):
            slp_profile_per_profile.loc[group.index] = [
                ("SLP", year, category)
            ] * group.shape[0]

        return slp_profile_per_profile

    def generate_samples(self, yearly_info_df, daily_info_df, nb_samples=None):
        profiles_to_use = pd.Series(index=yearly_info_df.index, dtype="object")
        years = yearly_info_df.index.str.slice(-5, -1).astype("int")
        for year, year_df in yearly_info_df.groupby(years):
            if year in self.slps:
                profiles_to_use.loc[year_df.index] = self.get_slp_to_use_for_year(
                    year, self.slps[year], year_df
                )
            elif year in self.rlps:
                profiles_to_use.loc[year_df.index] = [("RLP", year)] * year_df.shape[0]
            else:
                raise Exception(f"year {year} without slp or rlp")

        sampling_probs = []
        standard_p = np.array([1])
        category_to_index = dict(
            S21=0,
            S22=1,
            S11=2,
            S12=3,
        )
        for (meterID, date), info in daily_info_df.iterrows():
            yearly_consumption = yearly_info_df.at[meterID, "yearly_consumption"]
            slp_to_use = profiles_to_use.at[meterID]
            if len(slp_to_use) == 3:
                type, year, category = slp_to_use
                assert type == "SLP"
                slp_to_use = self.slps[year].iloc[:, category_to_index[category]]
            else:
                type, year = slp_to_use
                assert type == "RLP"
                slp_to_use = self.rlps[year]["RLPestU"]
            query_date = str(date.replace(year=int(meterID[-5:-1])).date())
            normalized_day = slp_to_use.loc[query_date].to_numpy().reshape((1, -1))
            day = normalized_day * yearly_consumption
            sampling_probs.append((day, standard_p))
        return sampling_probs
