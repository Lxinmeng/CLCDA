# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 07:43:04 2020

@author: OOO
"""
import numpy as np
import pandas as pd
from math import sqrt


pd.set_option('display.max_columns', None)


def readfile():
    fr = open('data\c-seq_final.fasta', 'r')
    seq = []
    for line in fr:
        seq.append(line)
    fr.close()
    rna_seq = seq
    return rna_seq


def pigment(rna_seq):
    x = []
    y = []
    pre_x = 0.5
    pre_y = 0.5
    size = 2 ** 3
    for i in rna_seq:
        if i == 'A':
            pre_x = pre_x * 0.5
            pre_y = pre_y * 0.5
            x.append(pre_x)
            y.append(pre_y)
            continue

        if i == 'T':
            pre_x = pre_x * 0.5 + 0.5
            pre_y = pre_y * 0.5
            x.append(pre_x)
            y.append(pre_y)
            continue

        if i == 'C':
            pre_x = pre_x * 0.5
            pre_y = pre_y * 0.5 + 0.5
            x.append(pre_x)
            y.append(pre_y)
            continue

        if i == 'G':
            pre_x = pre_x * 0.5 + 0.5
            pre_y = pre_y * 0.5 + 0.5
            x.append(pre_x)
            y.append(pre_y)
            continue

    vector_x = []
    vector_y = []
    vector_sum = []
    label = []

    for i in range(len(x)):
        label.append(int(x[i] * size) + int(y[i] * size) * size)
    #    print(label)
    for i in range(size * size):
        x_sum = 0
        y_sum = 0
        sum = 0
        for j in range(len(label)):

            if i == label[j]:
                x_sum = x_sum + x[j]
                y_sum = y_sum + y[j]
                sum += 1

        if sum > 0:
            x_sum = x_sum
            y_sum = y_sum
        else:
            x_sum = 0
            y_sum = 0

        vector_x.append(x_sum)
        vector_y.append(y_sum)
        vector_sum.append(sum)  #Numi

    vector = []
    for i in vector_sum:
        if i == 0:  # sum or i
            vector.append(0)
        else:
            z_score = (i - np.average(vector_sum)) / np.std(vector_sum)
            vector.append(z_score)   #Z
    for i in vector_x:
        vector.append(i)  #X
    for i in vector_y:
        vector.append(i)   #Y
    return vector


def P(a, b):
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    corr = s1.corr(s2)

    return corr



if __name__ == "__main__":

    #   rna_seq=['AC']
    rna_seq = readfile()
    j = 0
    rna_name = []
    seq_data = []
    rw = []

    for i in rna_seq:

        if i[1] != '':
            j = j + 1
        seq_data.append(pigment(i))

    k = 0
    total_seq = []
    for i in range(len(seq_data)):
        rw_part = []
        for j in range(len(seq_data)):
            if i != j:
                rw_part.append(P(seq_data[i], seq_data[j]))
            else:
                rw_part.append(1)
            k = k + 1
        total_seq.append(rw_part)


np.savetxt("result_data\c-seq_similarity.txt", total_seq)


