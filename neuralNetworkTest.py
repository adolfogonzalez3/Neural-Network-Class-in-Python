import dataProcessing as dp
import neuralNetwork as nn
import numpy as np
import random as rand
import itertools
import time

def binarize(X,flag = True):
	mean = np.mean(X)
	newCol = [0 if mean > c else 1 for c in X]
	#for c in X:
	#	if mean > c:
	#		newCol.append(0)
	#	else:
	#		newCol.append(1)
	return newCol

def min_max(X):
	X = np.array(X)
	xmin = X.min(axis=0)
	xmax = X.max(axis=0)
	tmp = (X - xmin)
	return (tmp/(xmax - xmin))
# this function makes the filters for the patients
# a filter is a list of 1's and 0's
# example: drug effectiveness = {0.2,0.6,NA} filter = {1,1,0}
def make_filter(x):
	filt = []
	for lab in x:
		newFilter = []
		for ele in lab:
			if ele == None:
				newFilter.append(0)
			else:
				newFilter.append(1)
		filt.append(newFilter)
	return filt
# changes all values of {-1,0,NA} to 0 
# changes all values of {1} to 1
def fix_labels(x):
	flag = False
	if len(np.shape(x)) == 1:
		x = [x]
		flag = True
	newLabels = []
	for lab in x:
		newLab = []
		for ele in lab:
			if ele == 1:
				newLab.append(1)
			else:
				newLab.append(0)
		newLabels.append(newLab)
	if flag == True:
		return newLabels[0]
	else:
		return newLabels

def remove_missing_rows(X,Y):
	newX = []
	newY = []
	for x,y in zip(X,Y):
		try:
			y = float(y)
		except (ValueError, TypeError):
			y = None
		if y != None:
			newX.append(x)
			newY.append(y)
	return newX,newY
def run_CV(X,Y,machine,epochs,experiments=1,CVFolds=10):
	results = {"accuracy":[[] for i in range(len(epochs))]}
	for exper in range(experiments):
		indices = [i for i in range(len(X))]
		rand.shuffle(indices)
		X = [X[index] for index in indices]
		Y = [Y[index] for index in indices]
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
				machine["training_run"](train["X"],train["Y"],ep)
				resultsPerExper["accuracy"][ind].append(machine["test_run"](testX,testY)*100)
				ind += 1
		for ind in range(len(epochs)):
			results["accuracy"][ind].append(np.mean(resultsPerExper["accuracy"][ind]))
	return {"accuracy":results["accuracy"]}

def get_dataSet(drugIndex):
	attr = [1] + [i for i in range(8,33)]
	cellLine = dp.read_csv("Cell.Line.Database.dkim.v1.0.csv",'@')
	cellLine = dp.get_subset(cellLine, attr)
	cellLine = dp.get_subset(cellLine,["Cell Line",cellLine["columns"][1+drugIndex]],indices = False,value='NA')
	summary1 = dp.read_csv("Summary_P1_UTA.csv")
	summary2 = dp.read_csv("Summary_P2_UTA.csv")
	summary3 = dp.read_csv("Summary_P3_UTA.csv")
	summary = dp.merge_data([summary1,summary2,summary3],"Cell Line")
	summary = dp.get_subset(summary,["Sypro_Ruby_P1","Sypro_Ruby_P2","Sypro Ruby_P3"],indices = False,remove=True)
	summaryGenes = summary["columns"]
	cellLineDrugs = cellLine["columns"]
	merged = dp.merge_data([summary,cellLine],"Cell Line")
	summary = dp.get_subset(merged,summaryGenes,indices = False)
	cellLine = dp.get_subset(merged,[cellLineDrugs[1]],indices = False,value='NA')
	return [[float(i) for i in x[1:]] for x in summary["matrix"]],[float(y[0]) for y in cellLine["matrix"]]

def main():
	shape = [55,150,150,1]#[55,20,20,1]
	dropoutProbs = [.6,.6,1]#[.6,.6,1]
	epochs = [100000]
	drugAcc = []
	Drugs = range(1)
	begin = time.clock()
	neuralNet = nn.neuralNet(shape)
	machine = {"training_run":lambda x,y,e: neuralNet.training_run(x,y,epochs=e,dropoutProb=dropoutProbs)
				,"test_run":lambda x,y: neuralNet.run(x,y)[1],"reset":neuralNet.reset,"print":neuralNet.get_weights}
	for drug in Drugs:#[Drugs]:
		print(drug)
		genes, drugEffectiveness = get_dataSet(drug)
		drugEffectiveness = binarize(drugEffectiveness)
		#genes = min_max(genes)
		results = run_CV(genes,drugEffectiveness,machine,CVFolds = 10,experiments=1,epochs=epochs)
		drugAcc.append(np.mean(results["accuracy"][-1]))
	#drugAcc = [run_CV2(genes,drugEffectiveness,machine,CV
	with open("results.csv","w") as csv:
		csv.write("Shape," + ":".join([str(i) for i in shape]) + "\n")
		csv.write("Epochs," + str(epochs) + "\n")
		csv.write("Dropout?," + str(all([i == 1 for i in dropoutProbs])) + "\n")
		csv.write("Dropout Probs," + ":".join([str(i) for i in dropoutProbs]) + "\n")
		
		for i in Drugs:
			csv.write("Drug " + str(i+1))
			if i != (len(Drugs)-1):
				csv.write(",")
		csv.write("\n")
		for acc in drugAcc:
			csv.write(str(acc))
			if acc is not drugAcc[-1]:
				csv.write(",")
		csv.write("\n")
	end = time.clock()
	print("Time taken: " + str(end-begin))
	print("Mean: " + str(np.mean(drugAcc)))

main()














