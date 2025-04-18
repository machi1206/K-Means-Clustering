# import stuff here
from src.K_Means import Point, KMeans
from pre_process import load_and_filter
from plot import plot_clusters_with_map, elbow_plot, silhouette_plot

# set the path to the geojson file you want to plot
geojson_file = 'src/districts.json'

# filter the data
telangana_data = load_and_filter(geojson_file)

# convert the data given to something we can implement K-Means to
dataset = [Point(row['Latitude'], row['Longitude']) for _, row in telangana_data.iterrows()]

kmeans = KMeans(k=3, change_threshold=0.001, max_iterations=5000, dataset=dataset)
centroids = kmeans.workflow()

# plot the data onto the map selected
plot_clusters_with_map(dataset, centroids, india_geojson=geojson_file)

# display the elbow plot
elbow_plot(telangana_data)

# display the silhouette plot
silhouette_plot(telangana_data)
