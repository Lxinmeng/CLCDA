import pandas as pd
import numpy as np

df = pd.read_excel("data\c-m_symbol.xlsx", header=None)
symbol = df.iloc[0].values
pf = pd.read_excel("data\c-m_symbol.xlsx")
total_data_list = []
for i in range(len(symbol)):
    features_list = []
    for j in range(len(symbol)):
        num = 0
        x = list(pf.values[:, i])
        y = list(pf.values[:, j])
        while np.nan in x:
            x.remove(np.nan)
        while np.nan in y:
            y.remove(np.nan)
        for itemx in x:
            for itemy in y:
                if itemx == itemy:
                    num = num+1
        x_score = num/len(x)
        y_score = num/len(y)
        final_score = max(x_score, y_score)
        features_list.append(final_score)
    total_data_list.append(features_list)
total_data_array = np.array(total_data_list)
np.savetxt("result_data\c-m_similarity1.txt", total_data_array)
print(1)







