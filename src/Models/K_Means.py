import random
from typing import Callable
from numpy import argmin, ndarray
import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import manhattan_distances


class K_Means(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            centroids: ndarray = None,
            clusters: int = 3,
            distance: Callable[[ndarray, ndarray | None], ndarray] = manhattan_distances,
            epochs: int = int(1E4),
            debug: bool = True,
            seed: int = random.randint(0, 1E6)
    ) -> None:
        super().__init__()
        self.centroids = centroids
        self.clusters = clusters
        self.distance = distance
        self.epochs = epochs
        self.debug = debug
        self.seed = seed

    def fit(self, X: ndarray):

        [points_number, features_number] = X.shape

        if (self.centroids is None) or (self.centroids.shape[0] != self.clusters):
            medoids_set = set()
            self.centroids = numpy.zeros([self.clusters, features_number])

            while len(medoids_set) < self.clusters:
                medoids_set.add(random.randint(0, points_number))
            
            for medoid_index, i in zip(medoids_set, self.clusters):
                self.centroids[i, : ] = X[medoid_index, :]
                
            
        current_epoch = 0
        work_finished = False

        while (current_epoch < self.epochs) and not work_finished:

            new_centroids = numpy.zeros([self.clusters, features_number])
            
            distances = self.distance(self.centroids, X)
            indices: ndarray = argmin(distances, axis=0).T

            for cluster_index in range(0, self.clusters):
                cluster_points = X[indices == cluster_index]

                try:
                    new_centroids[cluster_index, :] = cluster_points.mean(axis=0)
                except:
                    pass

            if (new_centroids == self.centroids).all():
                work_finished = True

            self.centroids = new_centroids
            current_epoch += 1

        return self


    def predict(self, X: ndarray):

        distances = self.distance(self.centroids, X)
        indices: ndarray = argmin(distances, axis=0).T

        return indices
        

    def score(self, X: ndarray):
        
        indices = self.predict(X)
        silhouette_coefficient = -1E9

        accumulator = 0

        for cluster_index in range(0, self.clusters):

            cluster_points = X[indices == cluster_index]
            [points_numbers, _] = cluster_points.shape
            cohesions = self.distance(cluster_points, cluster_points).mean(axis=0).reshape([-1, 1])

            separations : ndarray = numpy.zeros([points_numbers, self.clusters - 1])

            # Count other cluster
            examined_cluster = 0

            for other_cluster in range(0, self.clusters):

                if other_cluster == cluster_index:
                    continue
                
                other_points = X[indices == other_cluster]
                cluster_distances = self.distance(cluster_points, other_points)
                separations[:, examined_cluster] = cluster_distances.mean(axis=1)
                examined_cluster += 1

            min_sep_indices = argmin(separations, axis=1, keepdims=True)

            
            min_separations = numpy.take_along_axis(separations, min_sep_indices, axis=1)
            silhouette_vector : ndarray= 1 - (cohesions / min_separations)

            accumulator += silhouette_vector.sum(axis=0)

        silhouette_coefficient = accumulator / X.shape[0]

        print(f"[DEBUG] - n cluster {self.clusters} - {silhouette_coefficient}")
        return silhouette_coefficient







    