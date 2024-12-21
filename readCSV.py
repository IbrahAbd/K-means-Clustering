import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('basic1.csv')

xpoints = df['x']
ypoints = df['y']
colours = df['color']

data = list(zip(xpoints,ypoints))
#plt.figure(figsize=(8,6))
#scatter = plt.scatter(xpoints,ypoints,c=colours, cmap='viridis',s=50)
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Scatter plot')
#xpoints = np.array([df.x])
#ypoints = np.array([df.y])
#plt.plot(xpoints,ypoints)
#plt.show()


class KMeans:
    def __init__(self ,k ,max_iterations ,tolerance = 0.0001):
        # Number of Clusters
        self.k = k

        # Maximum iterations without convergence conditions being met.
        self.max_iterations = max_iterations

        # Tolerance to ensure that x,y pairs are different enough to each other to make an iteration valuable.
        self.tolerance = tolerance

    def fit(self,data):

        # Initialise the centroids randomly, do not let the same centroid be selected again (replace = False) upon iterations.
        self.centroids = data[np.random.choice(range(len(data)),self.k,replace = False)]

        for _ in range(self.max_iterations):

            # Initialise the clusters with empty lists e.g
            # self.clusters = {0: [], 1: [], 2: []}
            self.clusters = {i : [] for i in range(self.k)}

            for point in data:
                
                distances = [np.linalg.norm(point-centroid) for centroid in self.centroids]
                cluster = np.argmin(distances)
                self.clusters[cluster].append(point)