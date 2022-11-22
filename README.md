# 
**CLCDA: A novel Graph Convolution Network based framework for predicting disease-circRNA Associations.**

# Data description
* c-d_adj: interaction pairs between circRNA and disease.
* whole_feature.contents: features of circRNA and disease.
* c-m_symbol: interaction pairs between circRNA and mirRNA.
* c-rbp_symbol: interaction pairs between circRNA and RBP.
* c-seq_final: sequence of circRNA.
* circ_symbol: All circRNA's names.
* dis_symbol: All disease's names.
* symbol_all: Row name corresponding to the prediction matrix.
* disease_similarity_2076: similarity of disease.
* test.xlsx: Data Incremental test Set based on MNDR.
* test_mndr_neg_x.txt: Negative test set corresponding to each disease.
* test_neg.txt: Negative test set corresponding to all disease.


# Run steps
1.In folder similarity_calculation, run c-m_similarity.py,c-rbp_similarity.py and c-seq_similarity.py, 
  the results c-m_similarity.txt, c-rbp_similarity.txt, c-seq_similarity.txt are saved in folder result_data.
2.In folder similarity_calculation, run rwr.py,
  the results c-m_similarity_feature.txt, c-rbp_similarity_feature.txt, c-seq_similarity_feature.txt are saved in folder result_data.
3.In folder semi_auto, run net_embedding.py,
  the results c-m_features.txt, c-rbp_features.txt, c-seq_features.txt are saved in folder integration.
4.In folder similarity_calculation, run consine_similarity.py, 
  the result c_similarity_final.txt is saved in folder integration.
5.In folder similarity_calculation, run dis_similarity.py, 
  the result d_similarity.txt is saved in folder result_data.
6.In folder similarity_calculation, run adj_manage.py,
  the result c-d_adj.txt is saved in folder result_data.
7.In folder similarity_calculation, run feature_all.py,
  the result features_final.txt is saved in folder result_data.
8.In folder similarity_calculation, run rwr.py,
  the result whole_feature.txt is saved in folder result_data.
9.Run train.py to train the model and obtain the predicted scores for disease-circRNA associations,
  the prediction result adj_pre.txt is saved.

# Requirements
* Python 3.6.
* Pytorch
* numpy
* scipy
* sklearn

