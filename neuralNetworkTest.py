import dataProcessing as dp
import neuralNetwork as nn
import numpy as np
import random as rand
import itertools
import time
import matplotlib.pyplot as plt

def binarize(X):
	if len(np.shape(X)) > 1:
		mean = [np.mean(x) for x in np.transpose(X)]
		newCol = np.transpose([[0 if m > c else 1 for c in col] for col,m in zip(np.transpose(X),mean)])
	else:
		mean = np.mean(X)
		newCol = [0 if mean > c else 1 for c in X]
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
def fix_labels(X,Y,remove="NA"):
	newX = []
	newY = []
	for x,y in zip(X,Y):
		if y != remove:
			newX.append(x)
			newY.append(float(y))
	return np.array(newX),np.array(newY)
	

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
			train["Y"] = [ele for ele in itertools.chain(*[y for y in sets["Y"] if y is not testY])]
			ind = 0
			for ep in epochs:
				errorHist = machine["training_run"](train["X"],train["Y"],ep)
				resultsPerExper["accuracy"][ind].append(machine["test_run"](testX,testY)*100)
				ind += 1
		for ind in range(len(epochs)):
			results["accuracy"][ind].append(np.mean(resultsPerExper["accuracy"][ind]))
	results["errorHistory"] = errorHist
	return results

def create_files():
	ind = [i for i in range(1000)]
	dp.separate_file("Cell_line_RMA_proc_basalExp.csv","cell_line.csv",indices=ind,axis=1,delimiter="\t",newDelimiter="\t",transpose = True)
	dp.separate_file("v17_fitted_dose_response.csv","drug_effectiveness.csv",indices=[2,3,5],axis=0,delimiter=",",newDelimiter=",")
	
	
def get_dataSet():
	toSelect = {"DRUG_ID": ["1026"]}
	cellLine = dp.read_csv("cell_line.csv",delimiter="\t")
	cellLine["matrix"] = cellLine["matrix"][2:]
	cellLine["columns"].remove("GENE_SYMBOLS")
	cellLine["columns"] = ["COSMIC_ID"] + cellLine["columns"]
	cellLine["attributes"]["COSMIC_ID"] = cellLine["attributes"]["GENE_SYMBOLS"]
	cellLine["matrix"] = np.concatenate((np.transpose([[row[0].replace("DATA.","") for row in cellLine["matrix"]]]),[row[1:] for row in cellLine["matrix"]]),axis=1)
	drugEffectiveness = dp.read_csv("drug_effectiveness.csv",delimiter=",")
	drugEffectiveness = dp.select(drugEffectiveness,toSelect)
	drugEffectiveness = dp.get_subset(drugEffectiveness,["DRUG_ID"],indices=False,remove=True)
	cellLineAttr = cellLine["columns"]
	drugEffectivenessAttr = drugEffectiveness["columns"]
	merged = dp.merge_data([drugEffectiveness,cellLine],"COSMIC_ID")
	merged = dp.get_subset(merged,["COSMIC_ID"],indices=False,remove=True)
	cellLine = dp.get_subset(merged,cellLineAttr[1:],indices=False)
	drugEffectiveness = dp.get_subset(merged,[drugEffectivenessAttr[1]],indices=False)
	print(np.shape(drugEffectiveness["matrix"]))
	print(np.shape(cellLine["matrix"]))
	return np.array([[float(i) for i in x] for x in cellLine["matrix"]]), np.array([[float(i) for i in y] for y in drugEffectiveness["matrix"]])

	#attr = [1] + [i for i in range(8,33)]
	#cellLine = dp.read_csv("Cell.Line.Database.dkim.v1.0.csv",'@')
	#cellLine = dp.get_subset(cellLine, attr)
	#cellLine = dp.get_subset(cellLine,["Cell Line"] + cellLine["columns"][1:],indices = False)#,value='NA')
	#summary1 = dp.read_csv("Summary_P1_UTA.csv")
	#summary2 = dp.read_csv("Summary_P2_UTA.csv")
	#summary3 = dp.read_csv("Summary_P3_UTA.csv")
	#summary = dp.merge_data([summary1,summary2,summary3],"Cell Line")
	#summary = dp.get_subset(summary,["Sypro_Ruby_P1","Sypro_Ruby_P2","Sypro Ruby_P3"],indices = False,remove=True)
	#summaryGenes = summary["columns"]
	#cellLineDrugs = cellLine["columns"]
	#merged = dp.merge_data([summary,cellLine],"Cell Line")
	#summary = dp.get_subset(merged,summaryGenes,indices = False)
	#cellLine = dp.get_subset(merged,cellLineDrugs[1:],indices = False)#,value='NA')
	#return [[float(i) for i in x[1:]] for x in summary["matrix"]],[float(y[0]) for y in cellLine["matrix"]]
	#return np.array([[float(i) for i in x[1:]] for x in summary["matrix"]]), np.array([[i for i in y] for y in cellLine["matrix"]])


def main():
	begin = time.clock()
	genes, drugEffectiveness = get_dataSet()
	shape = [len(genes[0]),len(drugEffectiveness[0])]
	dropoutProbs = [1]
	epochs = [10000]
	drugAcc = []
	neuralNet = nn.neuralNet(shape,learningRate=0.01)
	machine = {"training_run":lambda x,y,e: neuralNet.training_run(x,y,epochs=e,dropoutProb=dropoutProbs)
				,"test_run":lambda x,y: neuralNet.run(x,y)[1],"reset":neuralNet.reset,"print":neuralNet.get_weights}
	Drugs = [1]
	genes = min_max(genes)
	#for drug in Drugs:
	#	print(drug)
	#	results = run_CV(genes,drugEffectiveness[:,drug],machine,CVFolds = 10,experiments=1,epochs=epochs)
	#	drugAcc.append(np.mean(results["accuracy"][-1]))
	drugAcc = []
	drugError = []
	for drug in Drugs:
		newX = genes
		newY = drugEffectiveness
		#newX, newY = fix_labels(genes,drugEffectiveness)
		newY = np.array(binarize(newY))
		results = run_CV(newX,newY,machine,CVFolds = 10,experiments=1,epochs=epochs)
		drugError.append(results["errorHistory"])
		drugAcc.append(np.mean(results["accuracy"][-1]))
	for err in drugError:
		plt.plot(np.arange(1,(len(err)+1),1),err)
		plt.show()
	
	with open("results.csv","w") as csv:
		csv.write("Shape," + ":".join([str(i) for i in shape]) + "\n")
		csv.write("Epochs," + str(epochs) + "\n")
		csv.write("Dropout?," + str(not all([i == 1 for i in dropoutProbs])) + "\n")
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


#create_files()
main()





