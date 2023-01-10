import abc
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


class BaseSampler(metaclass=abc.ABCMeta):
    """
    Abstract class for base samplers, these are simple single-level samplers
    and are used by the samplers in samplers.py to make more complex two-level samplers.
    """

    def __init__(self, info_preprocessing=None):
        self.info_preprocessing = info_preprocessing

    def preprocess_info_fit_transform(self, info):
        if self.info_preprocessing is None:
            return info
        return self.info_preprocessing.fit_transform(info)

    def preprocess_info_transform(self, info):
        if self.info_preprocessing is None:
            return info
        return self.info_preprocessing.transform(info)

    @abc.abstractmethod
    def fit(self, info, data):
        """
        Fit this sampler on the given info and data
        """
        raise NotImplementedError()

    def generate_samples(self, info, nb_samples):
        """
        Generate some samples of this fitted sampler based on info.
        For each row in info, a numpy array with length nb_samples of profile IDS is returned
        """
        sampling_probability_vectors = self.get_sampling_probabilities(info)

        samples = []
        for sample_probabilities in sampling_probability_vectors:
            sample = sample_probabilities.sample(
                nb_samples, replace=True, weights=sample_probabilities
            ).index.to_numpy()
            samples.append(sample)
        return samples

    @abc.abstractmethod
    def get_sampling_probabilities(self, info):
        """
        For each row in info, generate a pandas series with the sampling probabilities off different days in the
        training data_df
        """
        raise NotImplementedError()


class ClusteringBaseSampler(BaseSampler):
    """
    Abstract class for samplers that use a clustering to sample from
    (some more complex sampler classes need direct access to the clustering)

    """

    clustering: Union[pd.Series, None]

    @property
    def clustering_dict(self):
        """
        Simple helper function that makes it convenient to access all the members of a cluster with a certain index
        """
        return {
            index: df.index for index, df in self.clustering.groupby(self.clustering)
        }

    @abc.abstractmethod
    def get_cluster_probabilities(self, info):
        """
        For each row in info, return the probabilities that this info row belongs to a certain cluster
        Result is returned as a dataframe, index: same index as given info, columns: one column for each cluster, values: probability that a certain info row belongs to a certain cluster
        """
        raise NotImplementedError()

    def generate_samples(self, info, nb_samples):
        # add info preprocessing
        info = self.preprocess_info_transform(info)

        # get the clustering probabilities
        cluster_probabilities = self.get_cluster_probabilities(info)

        # get the clustering in a convenient format for sampling
        cluster_dict = self.clustering_dict

        samples_per_info_row = []
        for index in range(info.shape[0]):
            picked_clusters = np.random.choice(
                cluster_probabilities.shape[1],
                size=nb_samples,
                replace=True,
                p=cluster_probabilities[index, :],
            )
            cluster_idxs, counts = np.unique(picked_clusters, return_counts=True)
            samples = []
            for cluster_idx, count in zip(cluster_idxs, counts):
                samples.append(
                    cluster_dict[cluster_idx]
                    .sample(count, replace=True)
                    .index.to_numpy()
                )
            samples_per_info_row.append(np.concatenate(samples, axis=0))
        return samples_per_info_row


def cluster_probabilities_to_sample_probabilities(cluster_probabilities, clustering):
    """
    Converts from a probability to sample from each cluster to a probability to sample each sample
    based on a given clustering.
    """
    cluster_idx, counts = np.unique(clustering, return_counts=True)

    # sort both arrays based on cluster_idx
    sort_idxs = np.argsort(cluster_idx)
    cluster_idx, counts = cluster_idx[sort_idxs], counts[sort_idxs]

    # sample_prob_per_cluster[i] = probability that a random sample from cluster i gets sampled
    sample_prob_per_cluster = 1 / counts
    # for each sample in the dataset, P(sample selected| cluster of sample is selected)
    p_sample_selected_from_cluster = sample_prob_per_cluster[clustering]
    p_cluster_of_sample_selected = cluster_probabilities[:, clustering]
    p_sample_selected = p_sample_selected_from_cluster * p_cluster_of_sample_selected

    return p_sample_selected


class MetadataClusterSampler(ClusteringBaseSampler):
    """
    A class that represents an object that can sample from a clustering based on metadata

    Given some metadata, we assign the 'instance' to the closest centroid and sample from that cluster uniformly
    (instance because depending on the context that can be a day or a year)
    """

    def __init__(self, clusterer, info_preprocessing=None):
        super().__init__(info_preprocessing)
        self.clusterer = clusterer

        self.clustering = None
        self.cluster_centroids = None

    def fit(self, info, consumption_data):
        info = self.preprocess_info_fit_transform(info)

        # fit the clustering on the given metadata
        self.clusterer.fit(
            info,
        )
        self.clustering = pd.Series(self.clusterer.labels_, index=info.index)
        self.cluster_centroids = self.clusterer.cluster_centers_

    def get_cluster_probabilities(self, test_info):
        test_info = self.preprocess_info_transform(test_info)

        # assign the test_info to the closest centroids
        assigned_clusters = self._assign_profiles_to_clusters(test_info)

        # convert assigned clusters to cluster probabilities (assigned cluster: P = 1, other clusters: P = 0)
        # cluster_probabilities[i,j] = probability that cluster j is chosen for sample i (in this case only 0 and 1)
        cluster_probabilities = np.zeros(
            (test_info.shape[0], self.cluster_centroids.shape[0])
        )
        cluster_probabilities[range(test_info.shape[0]), assigned_clusters] = 1

        return cluster_probabilities

    def get_sampling_probabilities(self, test_info):
        test_info = self.preprocess_info_transform(test_info)

        # assign the test_info to the closest centroids
        assigned_clusters = self._assign_profiles_to_clusters(test_info)

        # for every cluster that appears make the sampling series
        non_zero_clusters = np.unique(assigned_clusters)
        clustering_dict = self.clustering_dict
        probability_series = dict()
        for cluster_idx in non_zero_clusters:
            profiles = clustering_dict[cluster_idx]
            probability_series[cluster_idx] = pd.Series(
                np.full(len(profiles), fill_value=1 / len(profiles)), index=profiles
            )

        sample_probs_per_test_info = []
        for cluster_idx, test_info_entry in zip(assigned_clusters, test_info.index):
            sample_probs_per_test_info.append(
                probability_series[cluster_idx].rename(test_info_entry)
            )

        return sample_probs_per_test_info

    def _assign_profiles_to_clusters(self, test_info):
        # distances between profiles_to_assign and centroids
        all_distances = euclidean_distances(test_info, self.cluster_centroids)

        # assign each profile to the closest cluster
        assigned_clusters = np.argmin(all_distances, axis=1)
        return assigned_clusters


class ConsumptionClusterSampler(ClusteringBaseSampler):
    """
    A class that represent an object that can sample from a clustering based on consumption data

    A probabilistic (or deterministic) classifier is used to learn to assign an 'instance' to the correct cluster based on exogenous attributes.

    This class can be used for yearly and for daily data (e.g. any other type of data as well)
    """

    def __init__(
        self,
        classifier,
        clusterer,
        info_preprocessing=None,
        deterministic=False,
        fillna=True,
    ):
        super().__init__(info_preprocessing)
        self.classifier = classifier
        self.clusterer = clusterer

        self.clustering = None
        self.clustering_lookup_dict = None
        self.deterministic = deterministic

        self.fillna = fillna

    def fit(self, info, consumption_data):
        # preprocess info if necessary
        info = self.preprocess_info_fit_transform(info)

        # save the consumption data for sampling
        if self.fillna:
            consumption_data = consumption_data.fillna(0)

        # fit the clustering on the consumption data
        self.clusterer.fit(
            consumption_data,
        )
        self.clustering = pd.Series(
            self.clusterer.labels_, index=consumption_data.index
        )
        self.clustering_lookup_dict = {
            label: df for label, df in self.clustering.groupby(self.clustering)
        }

        # fit the classifier to predict the cluster from metadata
        self.classifier.fit(info, self.clusterer.labels_)

    def clean(self):
        self.clusterer = None

    def get_cluster_probabilities(self, test_info):
        return self.classifier.predict_proba(test_info)

    def get_sampling_probabilities(self, test_info):
        # preprocess info if necessary
        test_info = self.preprocess_info_transform(test_info)

        # let the classifier predict the probabilities of being part of the cluster
        if self.deterministic:
            clusters = self.classifier.predict(test_info)
            nb_clusters = np.unique(self.clusterer.labels_).shape[0]
            cluster_probabilities = np.zeros((clusters.shape[0], nb_clusters))
            cluster_probabilities[np.arange(clusters.shape[0]), clusters] = 1
        else:
            cluster_probabilities = self.classifier.predict_proba(test_info)

        # calculate the sample probabilities when sampling uniformly from the chosen cluster
        sample_probabilities = cluster_probabilities_to_sample_probabilities(
            cluster_probabilities, self.clustering
        )
        sample_df = pd.DataFrame(
            sample_probabilities, index=test_info.index, columns=self.clustering.index
        )
        return list(item for _, item in sample_df.iterrows())

    def generate_samples(self, info, nb_samples):
        # preprocess info if necessary
        info = self.preprocess_info_transform(info)

        if self.deterministic:
            clusters = self.classifier.predict(info)
            nb_clusters = np.unique(self.clusterer.labels_).shape[0]
            cluster_probabilities = np.zeros((clusters.shape[0], nb_clusters))
            cluster_probabilities[np.arange(clusters.shape[0]), clusters] = 1
        else:
            cluster_probabilities = self.classifier.predict_proba(info)

        samples_per_day = []
        for cluster_probs in cluster_probabilities:
            picked_clusters = np.random.choice(
                np.arange(0, cluster_probs.shape[0]),
                nb_samples,
                replace=True,
                p=cluster_probs,
            )
            cluster_idxs, cluster_count = np.unique(picked_clusters, return_counts=True)
            samples_per_profile = []
            for cluster_idx, count in zip(cluster_idxs, cluster_count):
                samples_from_cluster = (
                    self.clustering_lookup_dict[cluster_idx]
                    .sample(count, replace=True)
                    .index.to_list()
                )
                samples_per_profile.extend(samples_from_cluster)
            samples_per_day.append(np.array(samples_per_profile))
        return samples_per_day


class ExpertDaySelectionSampler(BaseSampler):
    REQUIRED_COLUMNS = ["feelsLikeC", "isWeekend", "dayOfWeek", "month"]

    def __init__(self, allowed_temp_diff=2.5):
        # this class handles info preprocessing itself
        super().__init__()

        self.allowed_temp_diff = allowed_temp_diff

        self.info = None
        self.weekend_training_days = None
        self.weekday_training_days = None

    def fit(self, info, consumption_data):
        assert all(
            required_col in info.columns for required_col in self.REQUIRED_COLUMNS
        ), "FeelsLikeC should be in the info df"
        self.info = info.loc[:, self.REQUIRED_COLUMNS].assign(
            feelsLikeC=lambda x: x.feelsLikeC.astype("float")
        )

        # cache weekend/weekdays (will otherwise be calculated A LOT)
        is_weekend = self.info["isWeekend"] == 1.0
        self.weekend_training_days = self.info[is_weekend]
        self.weekday_training_days = self.info[~is_weekend]

    def _filter_based_on_feelslikec(self, df, temp):
        # filter based on T apparant (or C feels like in our weather df)
        min_t_threshold = temp - self.allowed_temp_diff
        max_t_threshold = temp + self.allowed_temp_diff
        matching_t_days = df.query(
            f"feelsLikeC >= {min_t_threshold} and feelsLikeC <= {max_t_threshold}"
        )
        return matching_t_days

    def _filter_based_on_daytype(self, df, day_of_week, month):
        query = f"dayOfWeek == {day_of_week} and month == {month}"
        return df.query(query)

    def get_sampling_probabilities(self, test_info):
        all_sampling_probs = []
        for (testMeterID, test_date), query_series in test_info.iterrows():
            # filter the training days based on the query_series
            if float(query_series.isWeekend) == 1:
                training_info = self.weekend_training_days
            else:
                training_info = self.weekday_training_days

            # filter based on FeelsLikeC
            matching_days = self._filter_based_on_feelslikec(
                training_info, float(query_series.feelsLikeC)
            )

            # if no similar weather days just use same weekday in same month
            if matching_days.shape[0] == 0:
                # print(query_series)
                # print(self.info.shape)
                matching_days = self._filter_based_on_daytype(
                    training_info, query_series.dayOfWeek, query_series.month
                )

            # if no similar days are found, just use all days from that month
            if matching_days.shape[0] == 0:
                matching_days = training_info.query(f"month == {query_series.month}")
            sampling_probs = pd.Series(
                np.full((matching_days.shape[0],), 1 / matching_days.shape[0]),
                index=matching_days.index,
            ).rename((testMeterID, test_date))
            all_sampling_probs.append(sampling_probs)
        return all_sampling_probs


class RandomBaseSampler(BaseSampler):
    """
    Base sampler that simply ignores all info and samples a number of random days
    Used in the ablation study for second-level random day selection.
    """

    def __init__(self, n_samples=100, random_state=None):
        # here info preprocessing doesn't matter at all
        super().__init__()
        self.n_samples = n_samples
        if random_state is not None:
            print("take care when parallelizing if you provide a seed!")
        self.bit_generator = np.random.default_rng(random_state).bit_generator
        self.consumption_data = None

        self.clustering = None

    def fit(self, info, daily_data):
        self.consumption_data = daily_data

    def get_sampling_probabilities(self, test_info):
        result = []
        samples_to_take = (
            self.consumption_data.shape[0]
            if self.n_samples is None
            else min(self.consumption_data.shape[0], self.n_samples)
        )
        for test_day in test_info.index:
            sample = self.consumption_data.sample(
                samples_to_take, random_state=self.bit_generator
            )
            probabilities = np.full((samples_to_take,), 1 / samples_to_take)
            probability_series = pd.Series(
                probabilities, index=sample.index, name=test_day
            )
            result.append(probability_series)
        return result
