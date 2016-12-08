import numpy as np
import time
import math

def binarize(X):
    if len(np.shape(X)) > 1:
        mean = [np.mean(x) for x in np.transpose(X)]
        newCol = np.transpose([[0 if m > c else 1 for c in col] for col,m in zip(np.transpose(X),mean)])
        #newCol = np.transpose([[0 if c > 0 else 1 for c in col] for col in np.transpose(X)])
    else:
        mean = np.mean(X)
        newCol = [[0 if mean > c else 1 for c in X]]
    return newCol

def min_max(X):
    X = np.array(X)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    tmp = (X - xmin)
    return np.divide(tmp,(xmax - xmin))

def rank_genes(X,Y):
    distrOneUp = np.array([0 for i in range(len(X[0]))])
    distrZeroDown = np.array([0 for i in range(len(X[0]))])
    total = 0
    for rowX,rowY in zip(X,Y):
        for y in rowY:
            for x,i in zip(rowX,range(len(rowX))):
                if x < 0.5:
                    if y == 0:
                        distrZeroDown[i] += 1
                    #else:
                    #    distrOneDown[i] += 1
                else:
                    #if y == 0:
                    #    distrZeroUp[i] += 1
                    #else:
                    if y != 0:
                        distrOneUp[i] += 1
                total += 1

    #mat = abs(np.divide(abs(distrZeroDown),len(X)) - np.divide(abs(distrOneUp),len(X)))
    mat = abs(np.divide(abs(distrZeroDown),total) - np.divide(abs(distrOneUp),total))
    mat = np.reshape(mat,[len(mat),1])
    mat = np.concatenate((mat,[[int(i)] for i in range(len(X[0]))]), axis = 1)
    mat = np.array(sorted(mat,key = lambda x: x[0], reverse = True))
    return [int(i) for i in mat[:,1]]

def signal_to_noise(X):
    X = [[np.float64(i) for i in row] for row in X]
    means = np.mean(X,axis=1)
    std = np.std(X,axis=1)
    snr = np.divide(means,std)
    snr = [(snr[i],i) for i in range(len(snr))]
    snr = np.array(sorted(snr,key = lambda x: x[0], reverse = True))
    return [int(i) for i in snr[:,1]]

def PCA(X):
    X = [[float(i) for i in row] for row in X]
    eig_vals, eig_vecs = np.linalg.eig(np.cov(np.transpose(X)))
    eig_pairs = [(i,j) for i,j in zip(eig_vals,eig_vecs)]
    eig_pairs = sorted(eig_pairs,key=lambda x: x[0],reverse = True)
    return np.hstack(tuple([np.reshape(pair[1],[len(pair[1]),1]) for pair in eig_pairs]))
