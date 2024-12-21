import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('basic1.csv')

xpoints = df['x']
ypoints = df['y']
colours = df['color']

data = list(zip(xpoints,ypoints))

class KMeans:
    def __init__(self ,k ,max_iterations):
        # Number of Clusters
        self.k = k

        # Maximum iterations without convergence conditions being met.
        self.max_iterations = max_iterations

    def fit(self,data):

        # Initialise the centroids randomly, do not let the same centroid be selected again (replace = False) upon iterations.
        self.centroids = data[np.random.choice(range(len(data)),self.k,replace = False)]

        for _ in range(self.max_iterations):

            # Initialise the clusters with empty lists e.g
            # self.clusters = {0: [], 1: [], 2: []}
            self.clusters = {i : [] for i in range(self.k)}

            for point in data:
                # Calculate the Euclidian distance from the point to every centroid and store the distances.
                distances = [np.linalg.norm(point-centroid) for centroid in self.centroids]

                # Select the smallest distance to a cluster and assign the point to that cluster.
                cluster = np.argmin(distances)
                self.clusters[cluster].append(point)


            new_centroids = []
            for cluster_points in self.clusters.values():
                # Check if cluster is empty
                if len(cluster_points) > 0:
                    # Calculate the mean point within the cluster.
                    mean_point = np.mean(cluster_points, axis=0)

                    # Find point within cluster closest to the mean.
                    closest_point = min(cluster_points , key=lambda p: np.linalg.norm(p - mean_point))

                    # Append closest point to centroid list which will be the new centroid.
                    new_centroids.append(closest_point)
                else:
                    # If cluster is empty, append the previous centroid.
                    new_centroids.append(self.centroids[len(new_centroids)])


            # Convergence condtion to break out of loop if no changes are made to the clusters and mean points.
            if np.array_equal(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

    # Assign points to centroids after model has been trained and centroids have been found.
    def predict(self, data):  
        predictions = []
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            cluster = np.argmin(distances)
            predictions.append(cluster)
        return predictions

# Convert data to numpy array for easier manipulation
points = df[['x', 'y']].to_numpy()

# Run K-Means with 3 clusters and maximum of 10 iterations.
kmeans = KMeans(k=3, max_iterations=10)
kmeans.fit(points)
df['cluster'] = kmeans.predict(points)


plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']

# Plot points and centroids on graph.
for cluster in range(kmeans.k):
    cluster_points = df[df['cluster'] == cluster]
    plt.scatter(cluster_points['x'], cluster_points['y'], c=colors[cluster], label=f'Cluster {cluster}')

# To mark the final centroid for visual demonstation.
for centroid in kmeans.centroids:
    plt.scatter(*centroid, c='black', marker='x', s=100, label='Centroid')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means for basic1.csv')
plt.show()
