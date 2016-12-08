import random as rand
import dataProcessing as dp
import itertools
import numpy  as np

# Runs cross validation on a machine learning model
# experiments determines how many k-fold experiments are done
# CVFolds determines how many folds for the cross validation
def run_CV(X,Y,machine,experiments=1,CVFolds=10):
	results = {"accuracy":[],"error_history":[]}
	for exper in range(experiments):
		indices = [i for i in range(len(X))]
		rand.shuffle(indices)
		X = X[indices]
		Y = Y[indices]
		sets = {"X":[],"Y":[]}
		sets["X"], sets["Y"] = dp.create_dataset(X,Y,CVFolds)
		resultsPerExper = {"accuracy": [],"error_history": []}
		for testX, testY in zip(sets["X"],sets["Y"]):
			machine["reset"]()
			train = {"X":[],"Y":[]}
			train["X"] = np.array(np.concatenate([x for x in sets["X"] if x is not testX]))
			train["Y"] = np.array([ele for ele in itertools.chain(*[y for y in sets["Y"] if y is not testY])])
			#testY = [[i] for i in testY]
			ind = 0
			resultsPerExper["error_history"].append(machine["training_run"](train["X"],train["Y"]))
			resultsPerExper["accuracy"].append(machine["test_run"](testX,testY))
		results["accuracy"].append(np.mean(resultsPerExper["accuracy"],axis=1))
		results["error_history"].append(resultsPerExper["error_history"])
	return results
