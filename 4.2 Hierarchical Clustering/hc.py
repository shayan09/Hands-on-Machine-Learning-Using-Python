# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#Using Dendrogram to find no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
Y_hc = hc.fit_predict(X)

#Visualising the Cluster
plt.scatter(X[Y_hc == 0, 0], X[Y_hc == 0, 1], s = 100, color='red', label='Careful')
plt.scatter(X[Y_hc == 1, 0], X[Y_hc == 1, 1], s = 100, color='blue', label='Standard')
plt.scatter(X[Y_hc == 2, 0], X[Y_hc == 2, 1], s = 100, color='green', label='Target')
plt.scatter(X[Y_hc == 3, 0], X[Y_hc == 3, 1], s = 100, color='yellow', label='Careless')
plt.scatter(X[Y_hc == 4, 0], X[Y_hc == 4, 1], s = 100, color='grey', label='Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()