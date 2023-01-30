from kneed import KneeLocator
import numpy as np
from tqdm import tqdm
import copy
import logging
import time


class ElbowMethod:
    """
        A decorator around a clusterer that selects the number of clusters automatically by running the clustering algorithm multiple times

    """
    def __init__(self, clusterer,  cluster_range, metric = None, nb_repeats = 1, show_progress = False):
        self.clusterer = clusterer
        self.metric = metric
        self.cluster_range = list(cluster_range)
        self.nb_repeats = nb_repeats

        self.locator = None

        self.n_clusters = None
        self.best_clusterer = None

        self.show_progress = show_progress


    def fit(self, data):
        logging.debug('Starting fit in elbow method')

        inertia_scores = np.zeros((len(self.cluster_range),), dtype='float')
        best_clusterers = np.zeros((len(self.cluster_range), ), dtype='object')
        if self.metric is not None:
            logging.debug('Calculating distance matrix')
            distance_matrix = self.metric(data)
            logging.debug(f"{round(distance_matrix.nbytes / 1024 / 1024,2)} used in distance matrix")
        else:
            distance_matrix = data

        iter = tqdm(self.cluster_range, desc = 'Elbow') if self.show_progress else self.cluster_range
        best_possible_inertia = None
        best_possible_clusterer = None
        for idx, nb_clusters in enumerate(iter):
            # special case for more instances than clusters
            if nb_clusters > data.shape[0]:
                nb_clusters = data.shape[0]
                # if this is the first time, make the best clustering
                if best_possible_inertia is None:
                    clusterer = copy.deepcopy(self.clusterer)
                    clusterer.n_clusters = nb_clusters
                    result = clusterer.fit(distance_matrix, )
                    best_possible_inertia = result.inertia_
                    best_possible_clusterer = clusterer

                # save the results of this best possible clustering
                inertia_scores[idx] = best_possible_inertia
                best_clusterers[idx] = best_possible_clusterer
                continue


            start_time = time.time()
            best_inertia = None
            best_clusterer = None
            for _ in range(self.nb_repeats):
                clusterer = copy.deepcopy(self.clusterer)
                clusterer.n_clusters = nb_clusters
                result = clusterer.fit(distance_matrix, )
                inertia = result.inertia_
                if best_inertia is None or inertia < best_inertia:
                    best_inertia = inertia
                    best_clusterer = clusterer
            inertia_scores[idx] = best_inertia
            best_clusterers[idx] = best_clusterer

            logging.debug(f'Iteration {idx}/{len(self.cluster_range)} took {time.time() - start_time} for {nb_clusters} clusters (best_inertia = {best_inertia:.2f})')


        locator = KneeLocator(self.cluster_range, inertia_scores, curve='convex', direction='decreasing')
        self.n_clusters = locator.knee

        if self.show_progress:
            self.locator = locator
        logging.debug(f'Found optimal number of clusters of {self.n_clusters}')

        if self.n_clusters is None:
            print(self.cluster_range)
            print(inertia_scores)
            # import matplotlib.pyplot as plt
            # plt.plot(self.cluster_range, inertia_scores)
            # plt.show()
            self.n_clusters = self.cluster_range[len(self.cluster_range)//2]
            print(f"No clear elbow found using {self.n_clusters} clusters")
            logging.debug(f'No clear elbow found! Using {self.n_clusters} clusters')
        best_nb_clusters_idx = self.cluster_range.index(self.n_clusters)
        self.best_clusterer = best_clusterers[best_nb_clusters_idx]
        return self

    @property
    def cluster_centers_(self):
        return self.best_clusterer.cluster_centers_

    @property
    def labels_(self):
        return self.best_clusterer.labels_

    def plot_knee(self):
        self.locator.plot_knee()