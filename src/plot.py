from shapely.geometry import MultiPoint
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from .K_Means import Point, KMeans, euclidean_distance
import geopandas as gpd

change_threshold = 0.001
max_iterations = 5000

def plot_clusters_with_map(dataset, centroids, telangana_geojson):
    # from the geojson files, we find the shape of the telangana district map and plot our data onto that map
    telangana_shape = gpd.read_file(telangana_geojson).to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    telangana_shape.plot(ax=ax, edgecolor='black', facecolor='none')

    # check for unique clusters and then plot each cluster with different colors
    seen = set()
    for point in dataset:
        label = None
        if point.cluster_id not in seen:
            label = f"Cluster {point.cluster_id}"
            seen.add(point.cluster_id)
        ax.scatter(point.longitude, point.latitude, c=f"C{point.cluster_id}", label=label) # plotting clusters

    for centroid in centroids: # plotting the centroids
        ax.scatter(centroid.longitude, centroid.latitude, color='black', marker='X', s=200, label="Centroid")

    add_cluster_borders(ax, dataset) # add borders to clusters to make it better for us to view the plot

    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend()

def plot_clusters(dataset, centroids=None): # plotting just the data on a coordinate map
        plt.figure(figsize=(10, 7))
        seen = set()
        for point in dataset:
            label = None
            if point.cluster_id not in seen:
                label = f"Cluster {point.cluster_id}"
                seen.add(point.cluster_id)
            plt.scatter(point.longitude, point.latitude, c=f"C{point.cluster_id}", label=label)
        for centroid in centroids:
            plt.scatter(centroid.longitude, centroid.latitude, color='black', marker='X', s=200, label="Centroid")
        plt.title("K-Means Clustering")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)

def elbow_plot(telangana_data): # calculate and make the elbow plot
    wcss_values = []
    K_range = range(1, 15) # range of K for our plot
    temp = None
    for k in K_range:
        dataset_copy = [Point(row['Latitude'], row['Longitude']) for _, row in telangana_data.iterrows()]
        kmeans = KMeans(k=k, change_threshold=change_threshold, max_iterations=max_iterations, dataset=dataset_copy)
        temp = kmeans.workflow()
        wcss = kmeans.compute_wcss()
        wcss_values.append(wcss)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss_values, 'bo-', linewidth=1, markersize=6)
    plt.title('Elbow Plot')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)

def silhouette_plot(telangana_data): # calculate and make the silhouette plot
    silho_values = []
    S_range = range(1, 15) # range of K for our plot
    for s in S_range:
        dataset_copy = [Point(row['Latitude'], row['Longitude']) for _, row in telangana_data.iterrows()]
        kmeans = KMeans(k=s, change_threshold=change_threshold, max_iterations=max_iterations, dataset=dataset_copy)
        kmeans.workflow()
        silho = kmeans.compute_silho()
        silho_values.append(silho)

    plt.figure(figsize=(8, 5))
    plt.plot(S_range, silho_values, 'bo-', linewidth=1, markersize=6)
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)


def add_cluster_borders(ax, dataset): # adding cluster borders to improve visibility
    cluster_groups = {}
    for point in dataset:
        cluster_groups.setdefault(point.cluster_id, []).append((point.longitude, point.latitude))

    patches = []
    for cluster_id, coords in cluster_groups.items():
        if len(coords) >= 3:
            polygon = MultiPoint(coords).convex_hull
            mpl_poly = MplPolygon(list(polygon.exterior.coords), closed=True)
            patches.append(mpl_poly)

    p = PatchCollection(patches, facecolor='none', edgecolor='black', linewidth=2, linestyle='--')
    ax.add_collection(p)
