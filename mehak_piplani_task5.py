import numpy as np
import json
import sys

edge_filename = sys.argv[1] 
output_file= sys.argv[2] 
with open(edge_filename) as f:
    jsonData = json.load(f)

f.close()

jsonData = np.array(jsonData)   

from sklearn.neighbors import kneighbors_graph

knn_graph = kneighbors_graph(jsonData, 38, mode='connectivity', include_self=False)
A = knn_graph.toarray()
A= A + A.T - np.diag(A.diagonal())


D = np.diag(A.sum(axis = 1))
laplacian_matrix=D-A
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
eigenvalues= np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

idx = np.argsort(eigenvalues)

for i in range(len(idx)):
    if eigenvalues[idx[i]] >= 0.00001:
        j = i
        break

train_data = []
k=2
for i in eigenvectors:
    node_embedding = []
    for q in range(0, k):
        node_embedding.append(i[j + q])

    train_data.append(node_embedding)
from sklearn.cluster import KMeans   
kmeans = KMeans(n_clusters=2,random_state=0).fit(train_data)


import matplotlib.pyplot as plt

plt.scatter(jsonData[:, 0], jsonData[:, 1], c=kmeans.labels_, cmap='viridis')
plt.savefig(output_file)