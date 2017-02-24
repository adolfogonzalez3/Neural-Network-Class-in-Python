"""Module contains methods to read in data"""
import numpy as np
import pandas as pd
import random as rand
import math

def read_csv(file_name, delimiter=","):
    """Reads in a file (Not needed)"""
    data = {"matrix": [], "attributes": {}, "columns": []}
    with open(file_name, "rt") as csv:
        flag = False
        for lines in csv:
            lines = lines.replace(delimiter+delimiter, delimiter+"NA"+delimiter)
            lines = lines.replace(delimiter+delimiter, delimiter+"NA"+delimiter)
            lines = lines.replace(delimiter+"\n", delimiter+"NA")
            lines = lines.split(delimiter)
            if flag is False:
                flag = True
                i = 0
                for ele in lines:
                    ele = ele.rstrip()
                    data["attributes"][ele] = i
                    data["columns"].append(ele)
                    i += 1
            else:
                mat = [i.rstrip() for i in lines]
                length_missing = len(data["columns"]) - len(mat)
                mat = mat + ['NA']*length_missing
                data["matrix"].append(np.array(mat))
    data["matrix"] = np.array(data["matrix"])
    return data

def merge_data(data_to_merge, identifier):
    """Not needed"""
    merged_data = {"matrix":[], "attributes": {}, "columns": []}
    master_row_index = {}
    row_indices = []
    identifier_exist_master = [rows[data_to_merge[0]["attributes"][identifier]] for rows in data_to_merge[0]["matrix"]]
    for data in data_to_merge[1:]:
        identifier_exist = [rows[data["attributes"][identifier]] for rows in data["matrix"]]
        identifier_exist_master = [existing_identifier for existing_identifier
                                   in identifier_exist_master
                                   if existing_identifier in identifier_exist]
    j = 0
    for i in identifier_exist_master:
        merged_data["matrix"].append([i])
        master_row_index[i] = j
        j += 1
    merged_data["attributes"][identifier] = 0
    merged_data["columns"].append(identifier)
    for data in data_to_merge:
        row_index = {}
        j = 0
        for row in data["matrix"]:
            IDOfRow = row[data["attributes"][ID]]
            if IDOfRow in IDExistMaster:
                rowIndex[IDOfRow] = j
            j += 1
        rowIndices.append(rowIndex)
    for data, indices in zip(dataToMerge, rowIndices):
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
            mergedData["matrix"][indexOfId] = np.add(
                mergedData["matrix"][indexOfId],
                list(np.array(data["matrix"][indices[existingID]])[toAdd]))
    return mergedData

def get_subset(data, attributes, remove=False, indices=True, value=None):
    newData = {"matrix": [], "attributes": {}, "columns": []}
    if indices == True:
        if remove == True:
            attributes = [attr for attr in data["columns"]
                          if data["attributes"][attr] not in attributes]
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
            if flag == True:
                newData["matrix"].append(newRow)
        j = 0
        for attr in attributes:
            newData["columns"].append(attr)
            newData["attributes"][attr] = j
            j += 1
    return newData

def create_dataset(X, Y, numberOfFolds):
    setsX = []
    setsY = []
    indices = range(len(Y))
    sampleSize = int(len(Y)/numberOfFolds)
    for folds in range(numberOfFolds):
        #if folds != numberOfFolds-1:
        setIndices = rand.sample(indices, sampleSize)
        indices = [index for index in indices if index not in setIndices]
        #else:
        #    setIndices = indices
        setsX.append(list(np.array(X)[setIndices]))
        setsY.append(list(np.array(Y)[setIndices]))
    for setX, setY, index in zip(setsX, setsY, indices):
        setX.append(np.array(X)[index])
        setY.append(np.array(Y)[index])
    return setsX, setsY


def fun(i):
    try:
        if math.isnan(i):
            return False
        else:
            return True
    except TypeError:
        return True

def get_datasetX():
    matX = pd.read_csv("dataX.csv", sep="\t", header=None)
    matX = matX.transpose()
    header = np.array(matX[:1])
    matX.columns = header[0]
    matX = matX[2:]
    total_rows = np.shape(matX)[1]
    drugs = [i for i in np.array(matX.columns) if fun(i)]
    matX = matX[drugs]
    return matX

def get_datasetY():
    matY = pd.read_csv("dataY.csv", sep=",")
    matY = matY[["COSMIC_ID", "DRUG_ID", "LN_IC50"]]
    matY.COSMIC_ID = [int(i) for i in matY["COSMIC_ID"]]
    return matY

def reduce_datasetY_by_drug(drug, matY):
    drug = np.array(list(set(matY["DRUG_ID"])))[drug]
    print(drug)
    result = matY[matY["DRUG_ID"] == drug]
    return result

def create_matrix_of_sensitivities(matY):
    cosmic_id = set(matY["COSMIC_ID"])
    drug_id = set(matY["DRUG_ID"])
    cosmic_matrix = [[tuple([i[1],i[2]]) for i in np.array(matY) if i[0] == j] for j in cosmic_id]
    iterator = 0
    diction = {}
    print("here")
    for drug in drug_id:
        diction.update({drug: iterator})
        iterator += 1
    matrix = np.empty([len(cosmic_id),len(drug_id)])
    matrix.fill(np.NAN)
    x_index = 0
    print("there")
    for cosmic in cosmic_matrix:
        for drug in cosmic:
            matrix[x_index][diction[drug[0]]] = drug[1]
        x_index += 1
    return matrix, cosmic_id, drug_id

def save_matrix(matrix, cosmic_id, drug_id):
    with open("matrixY.csv","wt") as csv:
        csv.write("Cosmic ID," + ",".join([str(i) for i in drug_id]) + "\n")
        for row, cosmic in zip(matrix,cosmic_id):
            csv.write(str(cosmic)+",")
            csv.write(",".join([str(i) for i in row]) + "\n")


def combine_datasets(matX, matY, flag=True):
    if flag is False:
        ID = pd.read_csv("Sorted_Patients_by_Type/SCLC.txt")
        matY = pd.merge(matY, ID, on="COSMIC_ID")
    genes = matX.columns[1:]
    mat = pd.merge(matX, matY, on="COSMIC_ID")
    return np.array(mat[genes]), np.array(mat[[matY.columns[2]]])

def combine_datasets2(matX, matY, flag=True):
    if flag is False:
        ID = pd.read_csv("Sorted_Patients_by_Type/SCLC.txt")
        matY = pd.merge(matY, ID, on="COSMIC_ID")
    genes = matX.columns[1:]
    mat = pd.merge(matX, matY, on="COSMIC_ID")
    return mat
