import abc
import copy
import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from scengen.models.basesamplers import (
    ClusteringBaseSampler,
    BaseSampler,
    ExpertDaySelectionSampler,
)


class Sampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_sampling_probabilities(self, yearly_info_df, daily_info_df):
        raise NotImplementedError()

    def generate_samples_and_convert_to_probs(
        self, yearly_info_df, daily_info_df, nb_samples
    ):
        """
        Method that calls generate_samples but converts the output to the same as get_sampling_probabilities
        """
        samples = self.generate_samples(yearly_info_df, daily_info_df, nb_samples)
        probs = []
        for sample, test_idx in zip(samples, daily_info_df.index):
            probabilities = pd.Series(
                np.full(sample.shape[0], fill_value=1 / sample.shape[0]),
                name=test_idx,
                index=sample,
            )
            probs.append(probabilities)
        return probs

    def generate_samples(self, yearly_info_df, daily_info_df, nb_samples):
        sampling_probability_vectors = self.get_sampling_probabilities(
            yearly_info_df, daily_info_df
        )

        samples = []
        for sample_probabilities in sampling_probability_vectors:
            sample = sample_probabilities.sample(
                nb_samples, replace=True, weights=sample_probabilities
            ).index.to_numpy()
            samples.append(sample)
        return samples

    @classmethod
    def check_fit_input(
        cls, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df
    ):
        assert (
            yearly_data_df.shape[0] == yearly_info_df.shape[0]
        ), "consumption data and yearly info should contain equal number of rows"
        assert yearly_data_df.index.equals(yearly_info_df.index), "indices should equal"
        assert daily_data_df.index.equals(daily_info_df.index)
        assert (
            len(
                daily_info_df.index.get_level_values(0).symmetric_difference(
                    yearly_data_df.index
                )
            )
            == 0
        ), "first level of daily_info df should contain the same values as the yearly consumption data and yearly_info_df"


class DailySamplerFromClusterSampler(Sampler):
    """
    IN USE
    Sampler that combines a yearly sampler with a daily sampler.
    Per cluster in the yearly sampler a daily sampler is learned.

    The yearly sampler is therefore assumed to produce a clustering of the yearly data.
    """

    def __init__(
        self,
        yearly_sampler: ClusteringBaseSampler,
        daily_sampler: BaseSampler,
        show_progress: bool = False,
    ):
        # daily and yearly sampler
        self.yearly_sampler: ClusteringBaseSampler = yearly_sampler
        self.daily_sampler_prototype: BaseSampler = daily_sampler

        # show progress of fitting the samplers or not
        self.show_progress: bool = show_progress

        # fitted daily sampler per cluster of the yearly sampler
        self.daily_sampler_per_cluster: Optional[Dict[int, BaseSampler]] = None

    def fit(self, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df):
        # check input
        self.check_fit_input(
            yearly_data_df, daily_data_df, yearly_info_df, daily_info_df
        )

        # fit the yearly sampler
        logging.debug(f"Fitting yearly sampler {self.yearly_sampler}")
        start_time = time.time()
        self.yearly_sampler.fit(yearly_info_df, yearly_data_df)
        clustering = self.yearly_sampler.clustering
        logging.debug(f"Fitting yearly sampler took {time.time() - start_time}")

        # fit the daily sampler for every yearly cluster
        logging.debug("Fitting the daily samplers")
        self.daily_sampler_per_cluster = dict()
        iter = clustering.groupby(clustering)

        # if necessary add tqdm around iterator to show progress
        if self.show_progress:
            iter = tqdm(iter, total=len(clustering.unique()), desc="Daily Cluster")

        # actual iteration
        for cluster_idx, cluster_df in iter:
            if len(cluster_df) == 0:
                continue
            # gather required info
            profiles_in_cluster = cluster_df.index
            days_in_cluster = daily_data_df.loc[profiles_in_cluster]
            day_info = daily_info_df.loc[profiles_in_cluster]

            # clone the prototype and train the daily sampler for this cluster
            logging.debug(
                f"Fitting daily sampler for cluster {cluster_idx} with {cluster_df.shape[0]} days."
            )
            start_time = time.time()
            cluster_sampler = copy.deepcopy(self.daily_sampler_prototype)
            cluster_sampler.fit(day_info, days_in_cluster)
            logging.debug(
                f"Daily sampler for cluster {cluster_idx} fit in {start_time - time.time()}"
            )

            # save sampler for later use
            self.daily_sampler_per_cluster[cluster_idx] = cluster_sampler

    def generate_samples(self, yearly_info_df, daily_info_df, nb_samples):
        # get the cluster probabilities for each unique household
        # cluster_probabilities_per_household is a dataframe
        # index all the unique household index and as columns the cluster indices
        # Values are the probability that a certain household belongs to a certain cluster
        cluster_probabilities_per_household = pd.DataFrame(
            self.yearly_sampler.get_cluster_probabilities(yearly_info_df),
            index=yearly_info_df.index,
        )

        # for each query day individually sample the days
        samples_per_day = []
        for (test_meterID, test_date), day_info in daily_info_df.iterrows():

            # gather the probabilities to belong to certain yearly clusters
            cluster_probs = cluster_probabilities_per_household.loc[
                [test_meterID], :
            ].iloc[0]

            picked_clusters = cluster_probs.sample(
                nb_samples, replace=True, weights=cluster_probs
            ).index
            cluster_idxs, cluster_count = np.unique(picked_clusters, return_counts=True)

            samples_for_profile = []
            for cluster_idx, count in zip(cluster_idxs, cluster_count):
                samples_from_cluster = self.daily_sampler_per_cluster[
                    cluster_idx
                ].generate_samples(day_info.to_frame().T, count)[0]
                samples_for_profile.extend(samples_from_cluster)
            samples_per_day.append(np.array(samples_for_profile))

        return samples_per_day

    def get_sampling_probabilities(self, yearly_info_df, daily_info_df):
        # get the cluster probabilities for each unique household
        # cluster_probabilities_per_household is a dataframe
        # index all the unique household index and as columns the cluster indices
        # Values are the probability that a certain household belongs to a certain cluster
        cluster_probabilities_per_household = pd.DataFrame(
            self.yearly_sampler.get_cluster_probabilities(yearly_info_df),
            index=yearly_info_df.index,
        )

        # for each query day individually sample the days
        sample_probabilities_per_day_collection = []
        for (test_meterID, test_date), day_info in daily_info_df.iterrows():

            # gather the probabilities to belong to certain yearly clusters
            cluster_probs = cluster_probabilities_per_household.loc[
                [test_meterID], :
            ].iloc[0]

            # for each non-zero cluster calculate the daily sampling probabilities using day_info
            daily_sampling_probs_for_profile = []
            (non_zero_clusters,) = np.where(cluster_probs > 0)
            for cluster in non_zero_clusters:
                daily_sampling_probs_in_cluster = self.daily_sampler_per_cluster[
                    cluster
                ].get_sampling_probabilities(day_info.to_frame().T)[0]

                # adjust for probability of picking this cluster
                daily_sampling_probs_in_cluster = (
                    daily_sampling_probs_in_cluster * cluster_probs[cluster]
                )

                # save the result of this cluster
                daily_sampling_probs_for_profile.append(daily_sampling_probs_in_cluster)

            # append the daily probabilities for all clusters of this test info and save the result
            daily_sampling_probs_for_profile = pd.concat(
                daily_sampling_probs_for_profile
            )
            sample_probabilities_per_day_collection.append(
                daily_sampling_probs_for_profile
            )

        return sample_probabilities_per_day_collection


class ExpertBasedFilterFromRandomYearSampler(Sampler):
    """
    Custom implementation of a specific sampler for the ablation study.
    This sampler uses the expert-based day selection sampler to select scenarios from a random year.
    So first-level year selection is random, second level day selection is expert-based
    """

    def __init__(self):
        self.day_filter = ExpertDaySelectionSampler()
        self.nb_years: Optional[int] = None

    def fit(self, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df):
        self.day_filter.fit(daily_info_df, None)
        self.nb_years = yearly_data_df.shape[0]

    def get_sampling_probabilities(self, yearly_info_df, daily_info_df):
        result = []
        sampling_probabilities = self.day_filter.get_sampling_probabilities(
            daily_info_df
        )
        # renormalize the probabilities such that you pick a random year first and then pick a random day from that year
        # to do this, ensure that probabilities of the same year sum up to 1/n
        for sample_series in sampling_probabilities:
            transformed = sample_series.groupby(
                level=0, axis=0, group_keys=False
            ).apply(lambda x: x / x.sum() / self.nb_years)
            result.append(transformed)
        return result


class RandomSampler(Sampler):
    """
    Sampler that simply ignores all info and just samples a fixed number of random days
    """

    def __init__(self, nb_days=None):
        self.nb_days: int = nb_days
        self.daily_data_df: Optional[pd.DataFrame] = None

    def fit(self, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df):
        self.daily_data_df = daily_data_df

    def generate_samples(self, yearly_info_df, daily_info_df, nb_samples):
        samples_per_day = []
        for _ in range(daily_info_df.shape[0]):
            samples_per_day.append(
                self.daily_data_df.sample(nb_samples, replace=True).index.to_numpy()
            )
        return samples_per_day

    def get_sampling_probabilities(self, yearly_info_df, daily_info_df):
        sample_probabilities_per_day = []
        for (meterID, date), query_series in daily_info_df.iterrows():
            sampled_days = self.daily_data_df.sample(self.nb_days).index
            prob_series = pd.Series(
                np.full((self.nb_days,), fill_value=1 / self.nb_days),
                index=sampled_days,
            ).rename((meterID, date))
            sample_probabilities_per_day.append(prob_series)
        return sample_probabilities_per_day
