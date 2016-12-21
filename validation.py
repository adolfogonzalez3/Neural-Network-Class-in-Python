"""This module contains functions for validating models"""
import random as rand
import itertools
import numpy as np
import data_processing as dp

# Runs cross validation on a machine learning model
# experiments determines how many k-fold experiments are done
# CVFolds determines how many folds for the cross validation
def run_cross_validation(data, labels, machine, experiments=1, number_of_cross_folds=10):
    """Run cross validation experiment and
    return accuracy, standard deviation, and error
    """
    results = {"accuracy": [], "error_history": [], "standard_deviation": []}
    for exper in range(experiments):
        indices = [i for i in range(len(data))]
        rand.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        sets = {"X": [], "Y": []}
        sets["X"], sets["Y"] = dp.create_dataset(data, labels, number_of_cross_folds)
        results_per_exper = {"accuracy": [], "error_history": []}
        for testX, testY in zip(sets["X"], sets["Y"]):
            machine["reset"]()
            train = {"X": [], "Y": []}
            train["X"] = np.array(np.concatenate([x for x in sets["X"] if x is not testX]))
            train["Y"] = np.array(
                [ele for ele in itertools.chain(*[y for y in sets["Y"] if y is not testY])])
            results_per_exper["error_history"].append(
                machine["training_run"](train["X"], train["Y"]))
            results_per_exper["accuracy"].append(machine["test_run"](testX, testY))
        results["accuracy"].append(np.mean(results_per_exper["accuracy"], axis=0))
        results["standard_deviation"].append(np.std(results_per_exper["accuracy"], axis=0))
        results["error_history"].append(results_per_exper["error_history"])
    return results
