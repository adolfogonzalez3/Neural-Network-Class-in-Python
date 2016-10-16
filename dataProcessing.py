import numpy as np
import random as rand

def separate_file(fileName,newName,indices,axis=0,delimiter=",",newDelimiter=",",transpose = False):
	matrix = []
	with open(fileName,"rt") as csv:
		for line,lineNumber in zip(csv,(i for i in range(1000000))):
			line = np.array(line.split(delimiter))
			if axis == 0:
				matrix.append([line[index].rstrip() for index in indices])
			else:
				if lineNumber in indices:
					matrix.append([ele.rstrip() for ele in line])
	if transpose == True:
		matrix = np.transpose(matrix)
	with open(newName,"w") as wrt:
		for line in matrix:
			wrt.write(newDelimiter.join(line) + "\n")
	

def read_csv(fileName,delimiter=",",missing=False):
	data = {"matrix": [], "attributes": {}, "columns": []}
	with open(fileName,"rt") as csv:
		flag = False
		for lines in csv:
			if missing == False:
				lines = lines.replace(delimiter+delimiter,delimiter+"NA"+delimiter)
				lines = lines.replace(delimiter+delimiter,delimiter+"NA"+delimiter)
				lines = lines.replace(delimiter+"\n",delimiter+"NA")
			lines = lines.split(delimiter)
			if flag == False:
				flag = True
				i = 0
				for ele in lines:
					ele = ele.rstrip()
					data["attributes"][ele] = i
					data["columns"].append(ele)
					i += 1
			else:
				mat = [i.rstrip() for i in lines]
				lengthMissing = len(data["columns"]) - len(mat)
				mat = mat + ['NA']*lengthMissing
				data["matrix"].append(np.array(mat))
	data["matrix"] = np.array(data["matrix"])
	return data

def merge_data(dataToMerge, ID):
	mergedData = {"matrix":[], "attributes": {}, "columns": []}
	masterRowIndex = {}
	rowIndices = []
	IDExistMaster = [rows[dataToMerge[0]["attributes"][ID]] for rows in dataToMerge[0]["matrix"]]
	for data in dataToMerge[1:]:
		IDExist = [rows[data["attributes"][ID]] for rows in data["matrix"]]
		IDExistMaster = [existingID for existingID in IDExistMaster if existingID in IDExist]
	j = 0
	for i in IDExistMaster:
		mergedData["matrix"].append([i])
		masterRowIndex[i] = j
		j += 1
	mergedData["attributes"][ID] = 0
	mergedData["columns"].append(ID)
	for data in dataToMerge:
		rowIndex = {}
		j = 0
		for row in data["matrix"]:
			IDOfRow = row[data["attributes"][ID]]
			if IDOfRow in IDExistMaster:
				rowIndex[IDOfRow] = j
			j += 1
		rowIndices.append(rowIndex)
	for data,indices in zip(dataToMerge,rowIndices):
		#IDIndex = data["attributes"]
		toAdd = []
		i = 0
		for attr in data["columns"]:
			j = len(mergedData["columns"])
			if attr not in mergedData["columns"]:
				toAdd.append(i)
				mergedData["columns"].append(attr)
				mergedData["attributes"][attr] = j
				j += 1
			i += 1
		for existingID in IDExistMaster:
			indexOfId = masterRowIndex[existingID]
			mergedData["matrix"][indexOfId] = mergedData["matrix"][indexOfId] + list(np.array(data["matrix"][indices[existingID]])[toAdd])
	return mergedData

def select(data,selected):
	newData = {"matrix": [], "attributes": {}, "columns": []}
	for row in data["matrix"]:
		newRow = []
		flag = True
		for ele,col in zip(row,data["columns"]):
			values = selected.get(col)
			if values != None:
				if ele in values:
					newRow.append(ele)
				else:
					flag = False
					break
			else:
				newRow.append(ele)
		if flag == True:
			newData["matrix"].append(newRow)
	newData["attributes"] = data["attributes"]
	newData["columns"] = data["columns"]
	return newData
				

def get_subset(data,attributes,remove = False, indices = True, value = None):
	newData = {"matrix": [], "attributes": {}, "columns": []}
	if indices == True:
		if remove == True:
			attributes = [attr for attr in data["columns"] if data["attributes"][attr] not in attributes]
		for rows in data["matrix"]:
			flag = True
			newRow = []
			for i in rows[attributes]:
				if i == value:
					flag = False
					break
				else:
					newRow.append(i)
			#newRow = [i for i in rows[attributes] if i != value]
			if flag == True:
				newData["matrix"].append(newRow)
		j = 0
		for attr in attributes:
			newData["columns"].append(data["columns"][attr])
			newData["attributes"][data["columns"][attr]] = j
			j += 1
	else:
		if remove == True:
			attributes = [attr for attr in data["columns"] if attr not in attributes]
		for rows in data["matrix"]:
			flag = True
			newRow = []
			for attr in attributes:
				index = data["attributes"][attr]
				if rows[index] == value:
					flag = False
					break
				else:
					newRow.append(rows[index])
			#newRow = [rows[data["attributes"][attr]] for attr in attributes if rows[data["attributes"][attr]] != value]
			if flag == True:
				newData["matrix"].append(newRow)
		j = 0
		for attr in attributes:
			newData["columns"].append(attr)
			newData["attributes"][attr] = j
			j += 1
	return newData
			
def create_dataset(X,Y,numberOfFolds):
	setsX = []
	setsY = []
	indices = range(len(Y))
	sampleSize = int(len(Y)/numberOfFolds)
	for folds in range(numberOfFolds):
		#if folds != numberOfFolds-1:
		setIndices = rand.sample(indices,sampleSize)
		indices = [index for index in indices if index not in setIndices]
		#else:
		#	setIndices = indices
		setsX.append(list(np.array(X)[setIndices]))
		setsY.append(list(np.array(Y)[setIndices]))
	for setX,setY, index in zip(setsX,setsY,indices):
		setX.append(np.array(X)[index])
		setY.append(np.array(Y)[index])
	return setsX,setsY
		

			
			
			
			
