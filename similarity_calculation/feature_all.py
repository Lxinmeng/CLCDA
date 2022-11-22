import numpy as np

"""
整合所有矩阵
"""

cd = np.loadtxt("D:\code\CDA\data\whole_feature\c-d_adj.txt")
cs = np.loadtxt("D:\code\CDA\similarity_calculation\semi_auto\data\output\integration\c_similarity_final.txt") #
ds = np.loadtxt("D:\code\CDA\similarity_calculation\\result_data\d_similarity.txt")

for i in range(len(cs)):
    for j in range(len(cs)):
        cd[i][j] = cs[i][j]

for i in range(913, len(ds)):
    for j in range(913, len(ds)):
        cd[i][j] = ds[i-913][j-913]

np.savetxt("result_data\\features_final.txt", cd)


print(1)