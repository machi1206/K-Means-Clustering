import numpy as np
import random

class Point:
    def __init__(self, latitude, longitude, cluster_id=None):
        self.latitude = latitude
        self.longitude = longitude
        self.cluster_id = cluster_id
        self.features = [self.latitude, self.longitude]

    def __repr__(self):
        return f"Point({round(self.latitude, 3)}, {round(self.longitude, 3)}) --> {self.cluster_id}"

class KMeans:
    def __init__(self, k, change_threshold, max_iterations, dataset, random_seed=None):
        self.k = k
        self.change_threshold = change_threshold
        self.max_iterations = max_iterations
        self.dataset = dataset
        self.random_seed = random_seed

    def assign_points(self, centroids):
        for point in self.dataset:
            point.cluster_id = self.find_cluster_id(centroids, point)

    def find_cluster_id(self, centroids, point):
        distances = [euclidean_distance(point.features, centroid.features) for centroid in centroids]
        return distances.index(min(distances))

    def find_new_centroid(self, cluster_id, centroids):
        points_in_cluster = [p for p in self.dataset if p.cluster_id == cluster_id]
        if not points_in_cluster:  
            return centroids[cluster_id]
        avg_lat = np.mean([p.latitude for p in points_in_cluster])
        avg_lon = np.mean([p.longitude for p in points_in_cluster])
        return Point(avg_lat, avg_lon, cluster_id)

    def update_centroids(self, centroids):
        total_change = 0
        new_centroids = []
        for i in range(self.k):
            new_centroid = self.find_new_centroid(i, centroids)
            old_centroid = centroids[i]
            change = euclidean_distance([old_centroid.latitude, old_centroid.longitude], [new_centroid.latitude, new_centroid.longitude])
            total_change += change
            new_centroids.append(new_centroid)
        return new_centroids, total_change / self.k

    def workflow(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        centroids = random.sample(self.dataset, self.k)
        change_in_cluster_assignments = float('inf')
        
        for _ in range(self.max_iterations):
            self.assign_points(centroids)
            centroids, change_in_cluster_assignments = self.update_centroids(centroids)
            if change_in_cluster_assignments < self.change_threshold:
                break
        
        self.centroids = centroids
        return centroids


    def compute_wcss(self):
        wcss = 0
        for point in self.dataset:
            centroid = self.centroids[point.cluster_id]
            distance = np.linalg.norm(np.array(point.features) - np.array(self.centroids[point.cluster_id].features))
            wcss += distance ** 2
        return wcss
    
    def compute_silho(self):
        silho_total = 0
        silho_count = 0

        for point1 in self.dataset:
            same_cluster_points = []
            other_clusters = {}

            for point2 in self.dataset:
                if point2.cluster_id == point1.cluster_id:
                    same_cluster_points.append(point2)
                else:
                    if point2.cluster_id not in other_clusters:
                        other_clusters[point2.cluster_id] = []
                    other_clusters[point2.cluster_id].append(point2)

            a = 0
            dists = [euclidean_distance(point1.features, p.features) for p in same_cluster_points if p != point1]
            a = sum(dists) / len(dists)

            b = float('inf')
            for cluster_points in other_clusters.values():
                dists = [euclidean_distance(point1.features, p.features) for p in cluster_points]
                avg_dist = sum(dists) / len(dists)
                if avg_dist < b:
                    b = avg_dist

            if max(a, b) > 0:
                s = (b-a)/max(a, b)
                silho_total += s
                silho_count += 1

        return silho_total / silho_count

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

