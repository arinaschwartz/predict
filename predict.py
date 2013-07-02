# CS121 A'11
# HW 7: predicting crime from calls to 311
#
# Arin Schwartz


def readFile(filename):
    r = open(filename, 'r')
    r = r.readlines()
    r = [x.strip().split(",") for x in r]
    return r
    

def extractFloatField(data, fieldNumber):
    floatList = []
    floatList.append(data[0][fieldNumber])
    [floatList.append(float(x[fieldNumber])) for x in data[1:]]
    return floatList

def listOfLists(list1, list2):
    listOfLists = []
    for j in range(1, len(list1)):
        sublist = [list1[j], list2[j]]
        listOfLists.append(sublist)
    return listOfLists

#Obtains predicted y value using linear regression
def predictedValues(model, listOfLists):
    predictedValues = []
    for j in listOfLists:
        for i in range(len(model)-1, 0, -1):
            
            predictedValues.append(model[0] + model[1] * j[0] + model[2] * j[1])
        return predictedValues

    

def singleVariable(data, i, j):
    r = readFile(data)
    x = extractFloatField(r, i)
    y = extractFloatField(r, j)
    LR = linear_regression(x[1:], y[1:])
    yp = []
    [yp.append((LR[1]*j)+LR[0]) for j in x[1:]]
    matplotlib.pyplot.scatter(x[1:], y[1:])
    matplotlib.pyplot.plot(x[1:], yp)
    
def saveFig(outputFilename):
    matplotlib.pyplot.savefig(outputFilename)
    
    
#single variable R2 calculation
def computeSingleVarR2(model, xs, ys):
    
    ys = ys[1:]
    xs = xs[1:]
    yp = []
    if len(model) == 2:
        [yp.append((model[1]*j)+model[0]) for j in xs]
        
    mean = 0.0
    for j in ys:
        mean = mean + j
    
    mean = mean/len(ys)
    
    error = 0.0
    
    for k in range(0, len(ys)):
        error = error + (ys[k] - yp[k])**2
    
    total = 0.0
    for i in ys:
        total = total + (i - mean)**2
    
    return (1-(error/total))


#UNIVERSAL R2 calculation
def computeR2(model, listOfLists, ys):
    if type(listOfLists[0]) != list:
        r2 = computeSingleVarR2(model, listOfLists, ys)
        return r2
    yp = predictedValues(model, listOfLists)
        
    mean = 0.0
    for j in ys:
        mean = mean + j
    
    mean = mean/len(ys)
    
    error = 0.0
    
    for k in range(0, len(ys)):
        error = error + (ys[k] - yp[k])**2
    
    total = 0.0
    for i in ys:
        total = total + (i - mean)**2
    
    return (1-(error/total))



#constructs a list-of-lists of r2 values, each sublist containing a column header and its corresponding r2 vs. total crime
def constructR2Table(data, LB, UB):
    f = readFile(data)
    r2Table = []
    for i in range(LB, UB):
        x = extractFloatField(f, i)
        y = extractFloatField(f, 19)
        r2Table.append([x[0], computeR2(linear_regression(x[1:], y[1:]), x, y)])
    return r2Table


def computeBivariateModel(data, xcols, y_col):
    f = readFile(data)
    y = extractFloatField(f, y_col)
    y = y[1:]
    models = []
    for i in xcols:
        for j in xcols[i:]:
            if j != i:
                list1 = extractFloatField(f, i)
                list2 = extractFloatField(f, j)
                x_list = listOfLists(list1, list2)
                LR = linear_regression(x_list, y)    for i in range(len(model)-1, 0, -1):
                models.append((LR, [i, j], computeR2(LR, x_list, y)))
    bestModel = models[0]
    for q in models:
        if q[2] > bestModel[2]:
            bestModel = q
    return bestModel
    
    

from pylab import *

########### Do NOT MODIFY THIS CODE ################
complaintCols = range(0,7)
complaintTotalCol = 18
crimeCols = range(8,18)
crimeTotalCol = 19

from numpy import ndarray, asarray, matrix, ones, hstack

def tall_and_skinny(A):
    return A.shape[0] > A.shape[1]

def prepend_ones_column(A):
    """
    Add a ones column to the left side of an array/matrix
    """
    ones_col = ones((A.shape[0], 1))
    return hstack((ones_col, A))

def linear_regression(X_input, Y_input):
    """
    Compute linear regression. Finds beta that minimizes
    X*beta - Y
    in a least squared sense.

    Accepts list-of-lists, arrays, or matrices as input type
    Will return an output of the same type as X_input

    Example:
    >>> X = matrix([[5,2], [3,2], [6,2.1], [7, 3]]) # covariates
    >>> Y = [5,2,6,6] # results - note that we can use either matrices or lists
    >>> beta = linear_regression(X, Y)
    >>> print beta
    [[ 1.20104895]
     [ 1.41083916]
     [-1.6958042 ]]
    >>> print prepend_ones_column(X)*beta # Note that the results are close to Y
    [[ 4.86363636]
     [ 2.04195804]
     [ 6.1048951 ]
     [ 5.98951049]]

    """
    # Convert any input into tall and skinny matrices
    X = matrix(X_input)
    Y = matrix(Y_input)
    if not tall_and_skinny(X):
        X = X.T
    if not tall_and_skinny(Y):
        Y = Y.T

    X = prepend_ones_column(X)

    # Do actual computation
    beta = (X.T*X).I * X.T * Y

    # Return type determined by X_input type
    if isinstance(X_input, list):
        return list(beta.flat)
    if isinstance(X_input, matrix):
        return beta
    if isinstance(X_input, ndarray):
        return asarray(beta.flat)
    raise NotImplementedError(
            "Expected input of list, matrix, or array, got %s"%beta.__class__)

######################


###  YOUR CODE GOES HERE
