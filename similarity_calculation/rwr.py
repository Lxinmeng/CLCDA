import numpy as np
from sklearn.preprocessing import minmax_scale

def _scaleSimMat(A):  #转移概率矩阵处理
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))  #对角线置0
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)   #每一列相加
    A = A.astype(np.float)/col[:, None]
    return A


def RWR(A, K=3, alpha=0.98):   #alpha重启概率 K：算法步数
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha * np.dot(P, A) + (1. - alpha) * P0
        M = M + P
    return M



network = np.loadtxt("result_data\c-seq_similarity.txt")
# network = np.loadtxt("result_data\\features_final")
network = RWR(network)
network = minmax_scale(network)
np.savetxt("result_data\c-seq_similarity_feature.txt", network)
# np.savetxt("result_data\whole_feature.txt", network)
print(1)