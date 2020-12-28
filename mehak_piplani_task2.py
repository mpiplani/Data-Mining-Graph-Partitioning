import json
import networkx as nx
import sys
import time
import numpy as np
G = nx.Graph() 
G.add_edge(1,3)
G.add_edge(2,3)
G.add_edge(3,8)
G.add_edge(3,5)
G.add_edge(4,5)
G.add_edge(5,6)
G.add_edge(7,8)
G.add_edge(8,9)
G.add_edge(8,10)
G.add_edge(10,11)
G.add_edge(10,12)

degree_seq = sorted(G.degree, key=lambda x: x[0])
degree_sequence_1 =[ i[1] for i in degree_seq]
A = nx.adjacency_matrix(G,nodelist=sorted(G.nodes))

laplacian_matrix = np.diag(degree_sequence_1) - A

eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
#eigenvectors=eigenvectors[:,idx]

for i in range(len(idx)):
    if eigenvalues[idx[i]] >= 0.00001:
        j = i
        break

train_data=[]

for i in eigenvectors:
    print(i)
    train_data.append(eigenvectors[i].tolist()[j:j+3])

   
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=4, random_state=0).fit(train_data)

community_0=[]
community_1=[]
community_2=[]
community_3=[]
for i,j in enumerate(kmeans.labels_):
    if j == 0:
        community_0.append(i+1)
    if j == 1:
        community_1.append(i+1)
    if j == 2:
        community_2.append(i+1)
    if j == 3:
        community_3.append(i+1)
print(community_0)
print(community_1)
print(community_2)
print(community_3)
with open('mehak_piplani_task2.txt', 'w') as fo:
    
    
    for i in train_data:
       
        temp = ""
        for j in i:
            
            temp = temp + str(j) + ","
           
        temp = temp[:-1]
        fo.write("%s\n" % temp)
    for i in kmeans.cluster_centers_:
       
        temp = ""
        for j in i.tolist():
            
            temp = temp + str(j) + ","
           
        temp = temp[:-1]
        fo.write("%s\n" % temp)
    
    

    temp = "{"
    for j in community_1:
        temp = temp + str(j) + ","
    temp = temp[:-1]
    temp = temp + "}"
    fo.write("%s," % temp)
    temp = "{"
    for j in community_0:
        temp = temp + str(j) + ","
    temp = temp[:-1]
    temp = temp + "}"
    fo.write("%s," % temp)
    temp = "{"
    for j in community_3:
        temp = temp + str(j) + ","
    temp = temp[:-1]
    temp = temp + "}"
    fo.write("%s," % temp)
    temp = "{"
    for j in community_2:
        temp = temp + str(j) + ","
    temp = temp[:-1]
    temp = temp + "}"
    fo.write("%s" % temp)
fo.close()
        
    