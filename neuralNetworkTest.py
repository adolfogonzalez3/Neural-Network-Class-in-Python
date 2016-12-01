import dataProcessing as dp
import neuralNetwork as nn
import numpy as np
import random as rand
import itertools
import time
import matplotlib.pyplot as plt
import preprocessing as pp


def run_CV(X,Y,machine,epochs,experiments=1,CVFolds=10):
	results = {"accuracy":[[] for i in range(len(epochs))]}
	errorHist = []
	print("CV")
	for exper in range(experiments):
		indices = [i for i in range(len(X))]
		rand.shuffle(indices)
		#X = [X[index] for index in indices]
		X = X[indices]
		#Y = [Y[index] for index in indices]
		Y = Y[indices]
		sets = {"X":[],"Y":[]}
		sets["X"], sets["Y"] = dp.create_dataset(X,Y,CVFolds)
		resultsPerExper = {"accuracy": [[] for i in range(len(epochs))]}
		for testX, testY in zip(sets["X"],sets["Y"]):
			machine["reset"]()
			train = {"X":[],"Y":[]}
			train["X"] = np.concatenate([x for x in sets["X"] if x is not testX])
			train["Y"] = [[ele] for ele in itertools.chain(*[y for y in sets["Y"] if y is not testY])]
			testY = [[i] for i in testY]
			ind = 0
			for ep in epochs:
				errorHist = machine["training_run"](train["X"],train["Y"],ep)
				resultsPerExper["accuracy"][ind].append(machine["test_run"](testX,testY)*100)
				ind += 1
		for ind in range(len(epochs)):
			results["accuracy"][ind].append(np.mean(resultsPerExper["accuracy"][ind]))
	results["errorHistory"] = errorHist
	return results

def get_dataSet():
	attr = [1] + [i for i in range(8,33)]
	cellLine = dp.read_csv("Cell.Line.Database.dkim.v1.0.csv",'@')
	cellLine = dp.get_subset(cellLine, attr)
	cellLine = dp.get_subset(cellLine,["Cell Line"] + cellLine["columns"][1:],indices = False)#,value='')
	summary1 = dp.read_csv("Summary_P1_UTA.csv")
	summary2 = dp.read_csv("Summary_P2_UTA.csv")
	summary3 = dp.read_csv("Summary_P3_UTA.csv")
	summary = dp.merge_data([summary1,summary2,summary3],"Cell Line")
	summary = dp.get_subset(summary,["Sypro_Ruby_P1","Sypro_Ruby_P2","Sypro Ruby_P3"],indices = False,remove=True)
	summaryGenes = summary["columns"]
	cellLineDrugs = cellLine["columns"]
	merged = dp.merge_data([summary,cellLine],"Cell Line")
	summary = dp.get_subset(merged,summaryGenes,indices = False)
	cellLine = dp.get_subset(merged,cellLineDrugs[1:],indices = False)#,value='NA')
	#return [[float(i) for i in x[1:]] for x in summary["matrix"]],[float(y[0]) for y in cellLine["matrix"]]
	return np.array([[float(i) for i in x[1:]] for x in summary["matrix"]]), np.array([[i for i in y] for y in cellLine["matrix"]])

def fun(x,y):
	newX, newY = fix_labels(x,y)
	newY = binarize(newY)
	return newX,newY

def main():
	X, Y = pp.get_dataset()
	X = pp.min_max(X)#[:,[2]]
	Y = pp.binarize(Y)
	rank = pp.rank_genes(X,Y)[:50]
	X = X[:,[int(i) for i in rank]]
	print(np.shape(X))
	print(np.shape(Y))
	size = int(len(X)/2)
	testX = X[-size:]
	testY = Y[-size:]
	trainX = X[:-size]
	trainY = Y[:-size]
	shape = [np.shape(X)[1],np.shape(Y)[1]]#[55,20,20,1]
	dropoutProbs = [1,1,1]#[.6,.6,1]
	epochs = [2000]
	begin = time.clock()
	neuralNet = nn.neuralNet(shape,learningRate=0.1,
		activations=["sigmoid"],lossFunction="squared_mean")
	machine = {"training_run":lambda x,y,e: neuralNet.training_run(x,y,epochs=e,dropoutProb=dropoutProbs)
				,"test_run":lambda x,y: neuralNet.run(x,y)[1],"reset":neuralNet.reset,"print":neuralNet.get_weights}
	machine["reset"]()

	errorHist = machine["training_run"](trainX,trainY,epochs[0])
	end = time.clock()
	accuracies = machine["test_run"](testX,testY)
	i = 1
	for acc in accuracies:
		print("Accuracy " + str(i) + ": " + str(acc))
		i += 1
	print("Total Accuracy: " + str(np.mean(accuracies)))
	print("Time taken: " + str(end-begin))
	print("Error History: " + str(errorHist))


main()
