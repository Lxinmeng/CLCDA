import numpy as np
import pandas as pd

"""
生成对称邻接矩阵
"""
df = pd.read_excel("D:\code\CDA\similarity_calculation\data\symbol_all.xlsx", header=None)
symbols = df.values[:, 0]
tmp = np.zeros((len(symbols), len(symbols)))


pf = pd.read_excel("D:\code\CDA\similarity_calculation\data\c_d_NEW.xlsx")

idx_features_labels = pf.values
# cir = pf.values[:, 0]
# dis = pf.values[:, 1]

for i in range(913):
        for data in idx_features_labels:
            if data[0] == symbols[i]:
                for j in range(913, len(symbols)):
                    if data[1] == symbols[j]:
                        tmp[i][j] = 1
for i in range(913, len(symbols)):
        for data in idx_features_labels:
            if data[1] == symbols[i]:
                for j in range(913):
                    if data[0] == symbols[j]:
                        tmp[i][j] = 1

np.savetxt("result_data\c-d_adj.txt", tmp)
print(1)





