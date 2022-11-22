from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import numpy as np

cs = np.loadtxt("similarity_caculation\semi_auto\data\output\integration\c-seq_features.txt") #c-seq
cm = np.loadtxt("similarity_caculation\semi_auto\data\output\integration\c-m_features.txt") #c-m
cr = np.loadtxt("similarity_caculation\semi_auto\data\output\integration\c-rbp_features.txt") #c-rbp
cf = np.concatenate((cm, cr, cs), axis=1)
s = cosine_similarity(cf)
network = minmax_scale(s)
np.savetxt("similarity_caculation\semi_auto\data\output\integration\c_similarity_final.txt", network)
print(1)

