import json
import sys
import time
import numpy as np
import sklearn.preprocessing 
filename = sys.argv[1]
output_file=sys.argv[2]
num_cluster=int(sys.argv[3])

file1 = open(filename,"r",encoding="utf-8")
lines=file1.readlines()
output_order=set()
matrix = [[0] * 1005 for i in range(1005)]
for i in lines:
    val = i.rstrip().split(" ")
    output_order.add(int(val[0]))
    output_order.add(int(val[1]))
    if int(val[0]) == int(val[1]):
        continue
        
    matrix[int(val[0])][int(val[1])]=1
    matrix[int(val[1])][int(val[0])]=1
    

D = np.diag(np.sum(matrix, axis=1))

laplacian_matrix=D-matrix

eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
j=0
for i in range(len(idx)):
    if eigenvalues[idx[i]] >= 0.00001:
        j = i
        break

train_data=[]
k = 3
for i in eigenvectors:
    node_embedding = []
    for q in range(0, k):
        node_embedding.append(i[j + q])

    train_data.append(node_embedding)
    
from sklearn.cluster import KMeans   
kmeans = KMeans(n_clusters=num_cluster).fit(train_data)

labels = kmeans.labels_
output_order=sorted(list(output_order))

outfile = open(output_file, "w+")
outstring=""

for i,j in enumerate(labels):
 
   
    outstring+=str(output_order[i])+' '+str(j)+'\n'   
   
outfile.write(outstring)
outfile.close()
