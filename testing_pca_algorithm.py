# -*- coding: utf-8 -*-

import cv2
import os
import numpy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from google.colab import drive

def calculateStandartDeviation(vector, targetMatrix):
    return numpy.apply_along_axis(calculateStandartDeviationHelper, 1, targetMatrix - vector)

def calculateStandartDeviationHelper(element):
    return numpy.average(element ** 2)

def calculateDispersion(vector, targetMatrix):
    return numpy.apply_along_axis(calculateDispersionHelper, 1, targetMatrix - vector)

def calculateDispersionHelper(element):
    return numpy.average(numpy.absolute(element))

def getId(vector, idTargetVector):
    return idTargetVector[numpy.argmin(vector)]

def getRatio(array1, array2):
    return str((len(array1) / 400) * 100) + '/' + str((len(array2) / 400) * 100)
     
def minMaxScaler(vector):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return min_max_scaler.fit_transform(vector) 

def normalize(vector):
    return preprocessing.normalize(vector)

def scale(vector):
    return preprocessing.scale(vector)

def plainDataset(vector):
    return vector

drive.mount('/content/drive', force_remount=True)
datasetFormatString = '/content/drive/My Drive/Colab Notebooks/dataset/s%d/%d.pgm'
allImages = []
allImagesId = []

for j in range(1, 11):
    for i in range(1, 41):
        img = cv2.imread(datasetFormatString % (i, j))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        allImages.append(gray.flatten())
        allImagesId.append(i)
print('images reshaping finished successful')

allImages = numpy.array(allImages, dtype = float)
calculationMethod = calculateStandartDeviation
pca = PCA(n_components=0.95)

normalizationMethods = [plainDataset, scale, normalize, minMaxScaler]

for normalizationMethod in normalizationMethods:
    nX = normalizationMethod(allImages)
    print(normalizationMethod.__name__)
    for ratio in range(1, 6):
        test_size = ratio / 10
        X_train, X_test, y_train, y_test = train_test_split(nX, allImagesId, test_size=test_size, random_state=1, shuffle=False)

        nMatrixWithDeviations = numpy.apply_along_axis(calculationMethod, 1, X_test, X_train)
        nVectorWithPredictedClasses = numpy.apply_along_axis(getId, 1, nMatrixWithDeviations, y_train)
        nDatasetResult = getRatio(X_train, X_test) + ' ' + '{0:.4f}'.format(sum(numpy.equal(nVectorWithPredictedClasses, y_test)) / len(y_test))

        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        matrixWithDeviations = numpy.apply_along_axis(calculationMethod, 1, X_test, X_train)
        vectorWithPredictedClasses = numpy.apply_along_axis(getId, 1, matrixWithDeviations, y_train)
        pcaDatasetResult = getRatio(X_train, X_test) + ' ' + '{0:.4f}'.format(sum(numpy.equal(vectorWithPredictedClasses, y_test)) / len(y_test))
        print(nDatasetResult, '  ', pca.n_components_ , ' ' , pcaDatasetResult)
    print(' ')
