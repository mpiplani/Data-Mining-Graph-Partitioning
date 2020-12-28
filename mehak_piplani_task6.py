import numpy as np
from collections import defaultdict
import sys

filename = sys.argv[1]
output_file = sys.argv[2] 

file1 = open(filename,"r",encoding="utf-8")
lines=file1.readlines()

matrix = [[0] * 1005 for i in range(1005)]
for i in lines:
    val = i.rstrip().split(" ")
    if int(val[0]) == int(val[1]):
       
        continue
        
    matrix[int(val[1])][int(val[0])]=1
    

D=np.sum(matrix,axis=0)

M = np.divide(matrix, D, out=np.ones_like(matrix)/1005, where=D != 0)

beta = 0.8

X = np.full((1005, 1005), (1 / 1005))

M = beta * M + (1 - beta) * X

eigenvalues, eigenvectors = np.linalg.eig(M)
eigenvalues=eigenvalues.real
eigenvectors=eigenvectors.real
j=0
for index,i in enumerate(eigenvalues):
    if np.isclose(i, 1):
        j = index
        break

nodes = eigenvectors[:, j].argsort()[::-1][:20]
outfile = open(output_file, "w+")
outstring=""
for i in nodes:
    outstring+=str(i)+'\n'   
   
outfile.write(outstring)
outfile.close()

