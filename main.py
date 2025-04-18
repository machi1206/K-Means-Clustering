# import stuff here
from src.K_Means import Point, KMeans
from src.pre_process import load_and_filter
from src.plot import plot_clusters_with_map, elbow_plot, silhouette_plot
import matplotlib.pyplot as plt

# set hyperparameters
change_threshold = 0.001
max_iterations = 5000

# geojson file of state of Telangana and its districts
geojson_file = 'src/districts.json'

# loading and filtering the data
telangana_data = load_and_filter(geojson_file)

# converting the data to a tangible object space
dataset = [Point(row['Latitude'], row['Longitude']) for _, row in telangana_data.iterrows()]

# workflow of algorithm
kmeans = KMeans(k=5, change_threshold=change_threshold, max_iterations=max_iterations, dataset=dataset)
centroids = kmeans.workflow()

# plotting data on the map selected
plot_clusters_with_map(dataset, centroids, telangana_geojson=geojson_file)

elbow_plot(telangana_data)

silhouette_plot(telangana_data)

plt.show()
