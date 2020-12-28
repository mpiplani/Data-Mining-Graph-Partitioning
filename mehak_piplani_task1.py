import json
import networkx as nx
import sys
import time
import numpy as np
"""
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
print(type(laplacian_matrix))"""
laplacian_matrix = [[1,0,-1,0,0,0,0,0,0,0,0,0],
                   [0,1,-1,0,0,0,0,0,0,0,0,0],
                   [-1,-1,4,0,-1,0,0,-1,0,0,0,0],
                   [0,0,0,1,-1,0,0,0,0,0,0,0],
                   [0,0,-1,-1,3,-1,0,0,0,0,0,0],
                   [0,0,0,0,-1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,-1,0,0,0,0],
                   [0,0,-1,0,0,0,-1,4,-1,-1,0,0],
                   [0,0,0,0,0,0,0,-1,1,0,0,0],
                   [0,0,0,0,0,0,0,-1,0,3,-1,-1],
                   [0,0,0,0,0,0,0,0,0,-1,1,0],
                   [0,0,0,0,0,0,0,0,0,-1,0,1]]

eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
eigenvalues= np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)
second_smallest_index= np.argsort(eigenvalues)[1]
second_smallest_value=eigenvectors[:,second_smallest_index]
print(second_smallest_value)

partition = [val >= 0 for val in eigenvectors[:, second_smallest_index]]

    # Compute the edges in between
nodes_in_A = [nodeA+1 for (nodeA, nodeCommunity) in enumerate(partition) if nodeCommunity]
nodes_in_B = [nodeB+1 for (nodeB, nodeCommunity) in enumerate(partition) if not nodeCommunity]
edges_in_between = []

"""       
laplacianMatrix = [[1,0,-1,0,0,0,0,0,0,0,0,0],
                   [0,1,-1,0,0,0,0,0,0,0,0,0],
                   [-1,-1,4,0,-1,0,0,-1,0,0,0,0],
                   [0,0,0,1,-1,0,0,0,0,0,0,0],
                   [0,0,-1,-1,3,-1,0,0,0,0,0,0],
                   [0,0,0,0,-1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,-1,0,0,0,0],
                   [0,0,-1,0,0,0,-1,4,-1,-1,0,0],
                   [0,0,0,0,0,0,0,-1,1,0,0,0],
                   [0,0,0,0,0,0,0,-1,0,3,-1,-1],
                   [0,0,0,0,0,0,0,0,0,-1,1,0],
                   [0,0,0,0,0,0,0,0,0,-1,0,1]]

with open('mehak_piplani_task1.txt', 'w') as fo:
    
    
    for i in laplacianMatrix:
       
        temp = ""
        for j in i:
            
            temp = temp + str(j) + ","
           
        temp = temp[:-1]
        fo.write("%s\n" % temp)

    temp = ""
    for j in second_smallest_value:
        
        temp = temp + str(j.item()) + ","
    temp = temp[:-1]
    fo.write("%s\n" % temp)

    temp = "{"
    for j in nodes_in_B:
        temp = temp + str(j) + ","
    temp = temp[:-1]
    temp = temp + "}"
    fo.write("%s," % temp)

    temp = "{"
    for j in nodes_in_A:
        temp = temp + str(j) + ","
    temp = temp[:-1]
    temp = temp + "}"
    fo.write("%s" % temp)

fo.close
"""