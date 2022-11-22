import numpy as np
import pandas as pd


idx_features_labels = np.loadtxt("D:\code\CDA\similarity_calculation\data\disease_similarity_2076.txt",dtype=str)
symbols = np.loadtxt("D:\code\CDA\similarity_calculation\data\dis_symbol.txt",dtype=str)
tmp = np.zeros((len(symbols), len(symbols)))
for i in range(len(symbols)):
        for data in idx_features_labels:
            if data[0] == symbols[i]:
                for j in range(len(symbols)):
                    if data[1] == symbols[j]:
                        tmp[i][j] = np.float64(data[2])
for i in range(len(symbols)):
        for data in idx_features_labels:
            if data[1] == symbols[i]:
                for j in range(len(symbols)):
                    if data[0] == symbols[j]:
                        tmp[i][j] = np.float64(data[2])
for i in range(len(symbols)):
    for j in range(len(symbols)):
        if i == j:
            tmp[i][j] = 1
            break
np.savetxt("result_data\d_similarity.txt", tmp)
print(1)





