"""This module contains functions for validating models"""
import random as rand
import itertools
import numpy as np
import data_processing as dp
import time
import Drug_Testing as dt


# Runs cross validation on a machine learning model
# experiments determines how many k-fold experiments are done
# CVFolds determines how many folds for the cross validation
def run_cross_validation(data, labels, machine, experiments=1, number_of_cross_folds=10, file_name=None):
    """Run cross validation experiment and
    return accuracy, standard deviation, and error
    """
    accuracies = []
    results = []
    for exper in range(experiments):
        indices = [i for i in range(len(data))]
        rand.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        sets = {"X": [], "Y": []}
        sets["X"], sets["Y"] = dp.create_dataset(data, labels, number_of_cross_folds)
        fold = 1
        for testX, testY in zip(sets["X"], sets["Y"]):
            begin = time.time()
            machine["reset"]()
            train = {"X": [], "Y": []}
            train["X"] = np.array(np.concatenate([x for x in sets["X"] if x is not testX]))
            train["Y"] = np.array(
                [ele for ele in itertools.chain(*[y for y in sets["Y"] if y is not testY])])
            #results_per_exper["error_history"].append(
            fold_name = file_name+"/"+"Fold_"+str(fold)
            print(fold_name)
            dt.make_sure_path_exists(file_name+"/")
            machine["training_run"](x=train["X"], y=train["Y"], a=testX, b=testY, f=fold_name)
            accuracy = machine["test_run"](testX, testY)
            accuracies.append(accuracy)
            end = time.time()
            results.append(tuple([fold,accuracy,(end-begin)]))
            print("Fold " + str(fold) + " done in " + str(end-begin) + " seconds.")
            fold += 1
    if file_name is not None:
        with open(file_name+"/"+"CV_Total.csv", "wt") as csv:
            csv.write("Fold,Accuracy,Time Taken\n")
            for row in results:
                csv.write(",".join([str(i) for i in row]) + "\n")
    return np.mean(accuracies), np.std(accuracies)

def run_bootstrap_validation(data,labels,machine,number_of_samples):
    """Bootstrap validation described in "A Study of Cross-Validation and Bootstrap
        for Accuracy Estimation and Model Selection"
        """
    results = list()
    sizeOfData = len(data)
    for sample in range(number_of_samples):
        sampleIndices = [rand.randint(0,sizeOfData-1) for i in range(sizeOfData)]
        testIndices = [i for i in range(sizeOfData) if i not in sampleIndices]
        testSetData = data[testIndices]
        testSetLabels = labels[testIndices]
        sampleSetData = data[sampleIndices]
        sampleSetLabels = labels[sampleIndices]
        machine["training_run"](sampleSetData, sampleSetLabels)
        epsilonZero = machine["test_run"](testSetData,testSetLabels)
        resub_acc = machine["test_run"](sampleSetData,sampleSetLabels)
        results.append(epsilonZero*.632+.368*resub_acc)
    return np.mean(results), np.std(results)
