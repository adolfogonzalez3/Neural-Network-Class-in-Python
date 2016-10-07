import openpyxl as xls
import matplotlib as plt
import numpy as np
import random as rand

def get_class_labels():
	# opens the excel file
	book = xls.load_workbook('Cell.Line.Database.dkim.v1.0.xlsm')#, read_only = True)
	# the table containing the colors used in the excel file
	COLOR_INDEX = xls.styles.colors.COLOR_INDEX
	# the main sheet
	sheet = book['Main']
	# the colors used
	GOOD = '#993366'
	NOEFFECT = '#C0C0C0'
	BAD = '#000080'
	rng = [i for i in range(8,33)]
	flag = False
	data = []
	dataOwner = []
	for rows in sheet.rows:
		if flag == True:
			dataOwner.append(rows[1].value.rstrip())
			dataRow = []
			for cells in [rows[i] for i in rng]:
				index = int(cells.fill.start_color.index)
				color = str('#'+COLOR_INDEX[index][2:])
				if color == GOOD:
					dataRow.append(1)
				elif color == NOEFFECT:
					dataRow.append(0)
				elif color == BAD:
					dataRow.append(-1)
				else:
					dataRow.append(None)
			data.append(dataRow)
		else:
			flag = True
	return dataOwner, data
	
def read_summary(summary):
	book = xls.load_workbook(summary)
	sheet = book[book.get_sheet_names()[0]]	
	flag = False
	data = []
	dataOwner = []
	for rows in sheet.rows:
		if flag == True:
			dataOwner.append(rows[0].value.rstrip())
			dataRow = []
			for cells in [rows[i] for i in range(1,len(rows))]:
				dataRow.append(cells.value)
			data.append(dataRow)
		else:
			flag = True
	return dataOwner, data

def read_csv(fileName, delimiter = ','):
	matrix = []
	with open(fileName,'rt') as csv:
		for lines in csv:
			row = []
			lines = lines.split(delimiter)
			for ele in lines:
				row.append(ele.rstrip())
			matrix.append(row)
	return matrix
	
def union_matrices(matrices, unionOn = None):
	if unionOn == None:
		unionOn = [0]*len(matrices)
	ID = []
	for mat,union in zip(matrices,unionOn):
		for row in mat:
			try:
				index = ID.index(row[union])
			except ValueError:
				index = None
			if index == None:
				ID.append(row[union])
	matrix = [[] for i in range(len(ID))]
	#print(ID)
	for mat,union in zip(matrices,unionOn):
		for row in mat:
			index = ID.index(row[union])
			for i in range(len(row)):
				if i != union:
					matrix[index].append(row[i])
	return ID, matrix
			
def lineup_matrices(matrices, lineOn = None):
	if lineOn == None:
		lineOn = [0]*len(matrices)
	newMatrices = [[] for i in range(len(matrices))]
	totalID = [[] for i in range(len(matrices))]
	for mat,lin,ID in zip(matrices,lineOn,totalID):
		for rows in mat:
			try:
				index = ID.index(rows[lin])
			except ValueError:
				index = None
			if index == None:
				ID.append(rows[lin])
	trueID = totalID[1]
	#print(totalID[1])
	#print(totalID[0])
	#input()
	#print(trueID)
	for ID in totalID:
		#print("what")
		trueID = [i for i in ID if i in trueID]
		#print(trueID)
	#input()
	#print(trueID)
	newMatrices = []
	for mat,lin in zip(matrices,lineOn):
		newMat = [[] for i in range(len(trueID))]
		for rows in mat:
			try:
				index = trueID.index(rows[lin])
				#print(index)
			except ValueError:
				index = None
			if index != None:
				newMat[index] = rows
		newMatrices.append(newMat)
	return ID, newMatrices


def read_cancer_data():
	summaries = ['Summary_P1_UTA.xlsx','Summary_P2_UTA.xlsx','Summary_P3_UTA.xlsx']
	labelId, labelClass = get_class_labels()
	summaryId = None
	summaryAttr = []
	
	for summ in summaries:
		a, b = read_summary(summ)
		summaryId = (a)
		summaryAttr.append(b)
	newSummaryAttr = np.concatenate((summaryAttr[0],summaryAttr[1],summaryAttr[2]), axis = 1)
	summaryAttr = newSummaryAttr
	newAttr = []
	newId = []
	newClass = []
	flag = False
	for i in range(len(summaryId)):
		flag = False
		for j in range(len(labelId)):
			if summaryId[i] == labelId[j]:
				newId.append(summaryId[i])
				newAttr.append(summaryAttr[i])
				newClass.append(labelClass[j])
				flag = True
				break		
	return newId, newAttr, newClass

def create_dataset(dataX,dataY,numberOfSets = 10):
	totalPatients = len(dataY)
	setSize = int(totalPatients/(numberOfSets))
	setsX = []
	setsY = []
	setOfElementsUsed = []
	setOfElementsNotUsed = [i for i in range(totalPatients)]
	
	for i in range(numberOfSets):
		if i == numberOfSets - 1:
			setSize = len(setOfElementsNotUsed)
		# randomly gets numbers, from 0 to totalPatients no repeating, for the
		# amount in trainingSize
		setIndices = rand.sample(setOfElementsNotUsed,setSize)
		# for every number in trainingSet
		newSetX = []
		newSetY = []
		for index in setIndices:
			newSetX.append(dataX[index])
			newSetY.append(dataY[index])
		setOfElementsUsed = setOfElementsUsed + setIndices
		setOfElementsNotUsed = [ele for ele in setOfElementsNotUsed if ele not in setOfElementsUsed]
		setsX.append(newSetX)
		setsY.append(newSetY)
	return setsX, setsY

def min_max(X):
	X = np.transpose(X)
	newX = []
	for col in X:
		maxi = max(col)
		mini = min(col)
		newCol = []
		for ele in col:
			val = ele - mini
			val = val/(maxi - mini)
			newCol.append(val)
		newX.append(newCol)
	return np.transpose(newX)
def zero_mean(X):
	X = np.transpose(X)
	newX = []
	for col in X:
		mean = np.mean(col)
		std = np.std(col)
		newCol = []
		for ele in col:
			val = ele - mean
			val = val/std
			newCol.append(val)
		newX.append(newCol)
	return np.transpose(newX)
			
		
		
#ID, X, Y = read_cancer_data()

#print(np.shape(X))
#print(np.shape(Y))










