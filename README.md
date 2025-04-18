# K-Means Clustering

This project implements the K-Means Clustering algorithm on a large dataset to demonstrate Unsupervised Machine Learning. It performs clustering on the data and provides insights into grouping similar data points.

We will be using only numpy, pandas, geopandas and, matplotlib libraries in the implementation.
## Program Implementation

### 1. Cloning the Repository
Clone the repository by running the following in your terminal,
```bash
git clone https://github.com/machi1206/K-Means-Clustering
```
This will make an exact copy of the entire repository onto your system to be able run locally. 
You might have to install git before running this command

After making a copy of the directory, navigate to folder with the code.
```bash
cd <enter_path_to_clone>
```

### 2. Creating an Environment
If you are running Python version 3.4 and above, venv is baked into the python installation and you can directly get started by 
```bash
python3 -m venv <enter_chosen_name_of_directory>
```
This will create a virtual environment in the directory you ran the command in with the name you specified above.

In order to activate the virtual environment, run the following
```bash
source <chosen_name_of_directory>/bin/activate
pip install -r requirements.txt
```
This will activate the virtual environment in the repo clone and then install all the necessary libraries required to run the program.

You will know if the virtual environment you created is activated if you see `(chosen_venv_name)` before your hostname `user@computer` in the terminal.

### 3. Execution of the Program
Run `main.py` to start the workflow of the project. 
```bash
python3 main.py
```
You can deactivate the virtual environment by running the following in the same directory the virtual environment is activated.
```bash
deactivate
```

## Program Inception
Our goal is to apply K-Means Clustering on the vast dataset we are provided. The very first step we take is filtering the given `.csv` file to the state of our choice. We have chosen Telangana as our sub-dataset.

One solution that we might think of implementing to filter the data and stick to Telangana might be checking the maximum and minimum latitudes and longitudes and hard-code the filters. Although it works well enough, there might be datapoints that are within the box of our constraints but outside the state. In order to solve this problem, we use the `geopandas` library and import a `GeoJSON` file corresponding to the boundary of Telangana and its districts.

![alt-text](https://github.com/machi1206/K-Means-Clustering/blob/main/plots/Filtered_Dataset.png)

We have now successfully filtered all of our data to fit our choice of state (Telangana). Our next step would just implementing the algorithm. The K-Means Algorithm initially assigns `K` number of centroids at random points within our bounds. Every point will then be assigned to the closest centroid and its cluster. Now that the assignment is done, we find a new centroid which is the coordinate mean of all points belonging to a given cluster.

This cyclic assignment and updating is done until certain conditions are met. If the centroids don't change by much (metric for this is `change_threshold`), then we stop the cycle and go ahead with the plotting and inference. If the `change_threshold` is too small, our compute time would be incredibly high so we introduce a new condition, `max_iterations`, this is basically the upper limit on how many times we want to run the cycle for any given `K` centroids.

![alt-text](https://github.com/machi1206/K-Means-Clustering/blob/main/plots/Flowchart.png)

Now that we have our clusters and their corresponding centroids, we can now plot them. I have defined 2 functions `plot_clusters_with_map` and `plot_clusters`. The first function will display the final cluster configuration onto a map of Telangana and their districts with a coordinate grid. The second function will plot just the data and clusters onto a longitude-latitude coordinate grid.

An important aspect to K-Means Clustering is finding the optimal value of `K`. We accomplish this using 2 different metrics. 

The most common way to get an optimal `K` value is through an Elbow Plot. The Elbow Plot is a graph between the WCSS (Within-Cluster Sum of Squares), which is the total distance between each point and its centroid in a given cluster, and the `K` value at which it was calculated. The optimal `K` value would be the point from whereon the elbow plot starts to gradually decrease. We basically want the point at which the change is maximum.

Below is the Elbow Plot for our filtered dataset. We can with reasonable confidence conclude that the optimal `K` value is 3.
![alt-text](https://github.com/machi1206/K-Means-Clustering/blob/main/plots/Elbow%20Plot.png)

Another way we can find the optimal `K` value is through a Silhouette Score Plot. The Silhouette Score basically measures how similar a given point is with its own cluster compared to other clusters. It ranges from -1 to 1. A higher score indicates well-defined solid clusters while a lower score indicates possible overlapping. A negative score says that the clustering is done incorrectly. The optimal `K` value would be the point with the highest Silhouette Score. 

Below is the Silhouette Score Plot of our filtered dataset. We can with reasonable confidence conclude that the optimal `K` value is 3 seeing that our Elbow Plot agrees with our Silhouette Plot.
![alt-text](https://github.com/machi1206/K-Means-Clustering/blob/main/plots/Silhouette%20Plot.png)

One would expect that the number of optimal clusters to be the number of districts in Telangana (which is 9) but we need to take into account that during our filtering process, we have discarded a lot of "useful" data which would have helped us get a better accurate clustering. 

Albeit `K=3` is optimal according to the data, it provides little to no information about the distribution of the dat. Upon checking for possible good candidates, we can also settle on `K=5`. It's position in the Elbow Plot is similar to that of `K=3` and its Silhouette Score is the second highest among all. It is also visually satisfactory. Upon further research, Telangana does seem to have 6 blocks based on cultural and logistic similarities which is slightly encouraging.

![alt-text](https://github.com/machi1206/K-Means-Clustering/blob/main/plots/K_equals_5.png)

## Program Analysis
### Structure
The repository has the following organization
```bash
K_Means_Clustering
|
|--- src
|    |--- __init__.py
|    |--- K_Means.py
|    |--- plot.py
|    |--- pre_process.py
|    |--- districts.json
|--- main.py
|--- clustering_data.csv
|--- requirements.txt
|--- README.md
|--- License
```
- `main.py` controls the entirety of the workflow of the program. change_threshold, max_iterations and, K are hyperparameters which we can set to our preferance.
- `clustering_data.csv` is the total given raw data from which we need to plot and infer from.
- `requirements.txt` is the list of all used libraries for the implementation of this project.
- `K_Means.py` is the utility module in which we have defined classes and functions necessary for the project.
- `plot.py` contains the necessary plotting and infering tools.
- `pre_process.py` loads and filters the data taken from `clustering_data.csv` and uses just the relevant datapoints for analysis.
- `__init__.py` makes the src directory a recognizable package. We cannot import objects from src module without the `__init__.py` file.

### Workflow
In `main.py`, the data is first filtered by state. Our analysis is limited to just the state of Telangana and its constituencies. We then filter out the data which don't have valid Latitudes and Longitudes. 

Our next task is to make sure the data we're considering for clustering is not out of state, we take the help of `geopandas` library and the `districts.json` file which contains the GeoJSON data for the various districts and the state border of Telangana.

Once we have filtered and cleaned our data, we create a database of `Point` objects inherited from the `Point` class. We then create a model `kmeans` from the `K_Means` class and give it the attributes we defined using the hyperparameters.

### Customisables
Our `change_threshold` variable decides the accuracy of the final centroid in our algorithm to the true centroid. Smaller the `change_threshold`, closer is the final centroid of a cluster to its true centroid. It basically dictates how small we want the change to be between consecutive centroids when we decide to stop running the algorithm.

Our `max_iterations` variable decides how deep we want our calculation to be. If we are willing to sacrifice speed and compute time for higher accuracy, we keep the `max_iterations` high.

We can specify the K value for K-Means and the `plot_clusters_with_map` function will display the final cluster groups after the algorithm's due course. 
We can also specify the range of scenarios we want to graph in our `elbow plot` and `silhouette plot`. 
For sake of universality and convenience, we have chosen our distance function to be Euclidean.



























