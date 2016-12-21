"""This module contains functions used to process data before
using for experiments
"""
import numpy as np

def binarize(data):
    """Gets the mean of a column and turns values less than the mean one
    otherwise zero
    """
    if len(np.shape(data)) > 1:
        means_list = [np.mean(column) for column in np.transpose(data)]
        new_data = np.transpose(
            [[0 if mean > element else 1 for element in column]
             for column, mean in zip(np.transpose(data), means_list)])
        #newCol = np.transpose([[0 if c > 0 else 1 for c in col] for col in np.transpose(X)])
    else:
        mean = np.mean(data)
        new_data = [[0 if mean > column else 1 for column in data]]
    return new_data

def min_max_data(data):
    """Processes the data with the min-max standarization function"""
    data = np.array(data)
    data_minimum = data.min(axis=0)
    data_maximum = data.max(axis=0)
    return (data - data_minimum)/(data_maximum - data_minimum)

def rank_genes(data, labels):
    """Ranks genes by the occurences of one label in respect to another"""
    distr_one_up = np.array([0 for element in range(len(data[0]))])
    distr_zero_down = np.array([0 for element in range(len(data[0]))])
    total = 0
    for data_row, label_row in zip(data, labels):
        for label in label_row:
            for element, i in zip(data_row, range(len(data_row))):
                if element < 0.5:
                    if label == 0:
                        distr_zero_down[i] += 1
                else:
                    if label != 0:
                        distr_one_up[i] += 1
                total += 1

    #mat = abs(np.divide(abs(distrZeroDown),len(X)) - np.divide(abs(distrOneUp),len(X)))
    mat = abs(abs(distr_zero_down)/total - abs(distr_one_up)/total)
    mat = np.reshape(mat, [len(mat), 1])
    mat = np.concatenate((mat, [[int(i)] for i in range(len(data[0]))]), axis=1)
    mat = np.array(sorted(mat, key=lambda x: data[0], reverse=True))
    return [int(i) for i in mat[:, 1]]

def use_signal_to_noise(data):
    """Uses the signal to noise function to rank the data"""
    data = [[np.float64(i) for i in row] for row in data]
    means = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    snr = means/std
    snr = [(snr[i], i) for i in range(len(snr))]
    snr = np.array(sorted(snr, key=lambda x: x[0], reverse=True))
    return [int(i) for i in snr[:, 1]]

def rank_using_pca(data):
    """Creates a matrix to transform the the data into a smaller representation"""
    data = [[float(i) for i in row] for row in data]
    eig_vals, eig_vecs = np.linalg.eig(np.cov(np.transpose(data)))
    eig_pairs = [(i, j) for i, j in zip(eig_vals, eig_vecs)]
    eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)
    return np.hstack(tuple([np.reshape(pair[1], [len(pair[1]), 1]) for pair in eig_pairs]))
