import dataProcessing as dp
import neuralNetwork as nn
import numpy as np
import random as rand
import itertools
import matplotlib.pyplot as plt
import time

def binarize(X,flag = True):
	mean = np.mean(X)
	newCol = []
	for c in X:
		if mean > c:
			newCol.append(0)
		else:
			newCol.append(1)
	return newCol

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
def run_CV2(X,Y,machine,epochs,experiments=1,CVFolds=10):
	results = {"accuracy":[[] for i in range(len(epochs))]}
	for exper in range(experiments):
		indices = [i for i in range(len(genes))]
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
			train["Y"] = fix_labels([[ele] for ele in itertools.chain(*[y for y in sets["Y"] if y is not testY])])
			testY = fix_labels([[i] for i in testY])
			ind = 0
			for ep in epochs:
				machine["training_run"](train["X"],train["Y"],ep)
				resultsPerExper["accuracy"][ind].append(machine["test_run"](testX,testY)*100)
				ind += 1
			#resultsPerExper["accuracy"].append( machine["test_run"](testX,testY)*100 )
		for ind in range(len(epochs)):
			results["accuracy"][ind].append(np.mean(resultsPerExper["accuracy"][ind]))
	return {"accuracy":results["accuracy"]}
		
def run_CV(X,Y):
	#if options == None:
	#	options = {"NUMOFEPOCH":1000,
	#####
	# create the data sets
	#####
	neuralNet = nn.neuralNet([len(X[0]),1])
	patientTotalDrugAcc = []
	setsX, setsY = dp.create_dataset(X,Y,10)
	cv = 0
	for testX,testY,CV in zip(setsX,setsY,range(len(setsY))):
		accuracyPerEpoch = []
		fitPerEpoch = []
		neuralNet.reset()
		#print(neuralNet.get_weights())
		#input()
		#####
		#
		#####
		trainX = np.concatenate([setsX[x] for x in range(len(setsX)) if x != CV])
		trainY = fix_labels([[ele] for ele in itertools.chain(*[setsY[y] for y in range(len(setsY)) if y != CV ])])
		testY = fix_labels([[i] for i in testY])
		#####
		#
		#####
		NUMOFEPOCHS = 1000
		#fit, acc = neuralNet.training_run(trainX,trainY,dropoutProb=[1,.3,1])
		neuralNet.training_run(trainX,trainY,epochs = NUMOFEPOCHS)
		#####
		#
		#####
		y = neuralNet.run(testX,testY)[1]
		#####
		cv += 1
		patientTotalDrugAcc.append(100*y)
	
	return {"Mean":np.mean(patientTotalDrugAcc),"Standard Deviation":np.std(patientTotalDrugAcc)}

def get_dataSet(drugIndex):
	attr = [1] + [i for i in range(8,33)]
	cellLine = dp.read_csv("Cell.Line.Database.dkim.v1.0.csv",';')
	cellLine = list([np.array(rows)[attr] for rows in cellLine])
	summary1 = dp.read_csv("Summary_P1_UTA.csv")
	summary1 = [[summ[i] for i in range(len(summ)) if i != 1] for summ in summary1 if summ is not summary1[0]]
	summary2 = dp.read_csv("Summary_P2_UTA.csv")
	summary2 = [[summ[i] for i in range(len(summ)) if i != 1] for summ in summary2 if summ is not summary2[0]]
	summary3 = dp.read_csv("Summary_P3_UTA.csv")
	summary3 = [[summ[i] for i in range(len(summ)) if i != 1] for summ in summary3 if summ is not summary3[0]]
	ID, summary = dp.union_matrices([summary1,summary2,summary3])
	ID = [[i] for i in ID]
	summary = np.concatenate((ID,summary),axis=1)
	ID, matrices = dp.lineup_matrices([summary,cellLine],[0,0])
	genes, drugEffectiveness = matrices
	genes = [g[1:] for g in genes]
	drugEffectiveness = [d[1:] for d in drugEffectiveness]
	drugEffectiveness = [drug[drugIndex] for drug in drugEffectiveness]
	genes, drugEffectiveness = remove_missing_rows(genes,drugEffectiveness)
	genes = [[float(i) for i in g] for g in genes]
	drugEffectiveness = [float(d) for d in drugEffectiveness]
	drugEffectiveness = binarize(drugEffectiveness)
	genes = dp.min_max(genes)
	drugEffectiveness = fix_labels(drugEffectiveness)
	return genes, drugEffectiveness


with open("results.csv","w") as csv:
	csv.write("Input nodes = 56,output node = 1\n")
	for drug in range(25):
		print(drug)
		genes, drugEffectiveness = get_dataSet(drug)
		csv.write("Drug: " + str(drug + 1) + "\n")
		csv.write("Number of Cell Lines = " + str(len(drugEffectiveness)) + "\n")
		epochs = [100]*10
		begin = time.clock()
		neuralNet = nn.neuralNet([len(genes[0]),1])
		machine = {"training_run":lambda x,y,e: neuralNet.training_run(x,y,epochs=e,dropoutProb=[1,.3,1]),"test_run":lambda x,y: neuralNet.run(x,y)[1],"reset":neuralNet.reset}
		results = run_CV2(genes,drugEffectiveness,machine,CVFolds = 10,experiments=10,epochs=epochs)
		end = time.clock()
		csv.write("Time elapsed: " + str(end-begin) + "\n")
		csv.write("Epochs,")
		for i in range(len(epochs)):
			csv.write(str((i+1)*epochs[0]))
			if i != (len(epochs) - 1):
				csv.write(",")
		csv.write("\nAccuracy,")
		for acc in results["accuracy"]:
			csv.write(str(np.mean(acc)))
			if acc is not results["accuracy"][-1]:
				csv.write(",")
		csv.write("\nStd,")
		for sd in results["accuracy"]:
			csv.write(str(np.std(sd)))
			if sd is not results["accuracy"][-1]:
				csv.write(",")
		csv.write("\n")
















