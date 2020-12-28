import json
import sys
import time
import numpy as np
import sklearn.preprocessing 
import pandas as pd
import csv
   
edge_filename = sys.argv[1]
train_filename=sys.argv[2]
test_filename=sys.argv[3]
output_file=sys.argv[4]


file_2=pd.read_csv(train_filename,encoding="utf-8",header=None)
train_label=[]
train_index=[]
for index,i in file_2.iterrows():
    train_label.append(int(i[0].split(" ")[1]))
    train_index.append(int(i[0].split(" ")[0]))
 
file1 = open(edge_filename,"r",encoding="utf-8")
lines=file1.readlines()
matrix = [[0] * 1005 for i in range(1005)]
for i in lines:
    val = i.rstrip().split(" ")
    if int(val[0]) == int(val[1]):
        continue
    matrix[int(val[0])][int(val[1])]=1
    matrix[int(val[1])][int(val[0])]=1
   
D = np.diag(np.sum(matrix, axis=1))

laplacian_matrix=D-matrix

eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
#eigenvalues= np.real(eigenvalues)
#eigenvectors = np.real(eigenvectors)
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
#eigenvectors=eigenvectors[:,idx]
j=0
for i in range(len(idx)):
    if eigenvalues[idx[i]] >= 0.00001:
        j = i
        break

train_data=[]
test_data={}
k = 76
for index,i in enumerate(eigenvectors):
    node_embedding = []
    for q in range(0, k):
        
        node_embedding.append(i[j + q])
    
    if index in train_index:
        train_data.append(node_embedding)
    else:
        test_data[index]=node_embedding


#train_data=sklearn.preprocessing.normalize(train_data)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(train_data,train_label)
file_3=pd.read_csv(test_filename, header=None,encoding="utf-8")
test_file=[]
test_index=[]
for index,i in file_3.iterrows():
    test_index.append(int(i[0].split(" ")[0]))
    test_file.append(test_data[int(i[0].split(" ")[0])])

predicted=neigh.predict(test_file)

   

output_string=[]
for i in range(len(test_index)):
    output_string.append([str(test_index[i])+" "+str(predicted[i])])

pd.DataFrame(output_string).to_csv(output_file,encoding='utf-8',index=False,header=None)
groundTruth = []

with open("data/labels_test_truth.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=" ", quotechar='"')
    for line in reader:
        key = int(line[0])
        value = int(line[1])
        groundTruth.append(value)

f.close()

from sklearn.metrics import accuracy_score

a = accuracy_score(groundTruth, predicted)


print(a)