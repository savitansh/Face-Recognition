"""
=======================================================
Perform face recognition using covariance based eigenface 
decomposition and classification using training set
=======================================================


EigenValue based decomposition of covariance matrix applied to sample data
to obtaing lower dimension feature space or eigenfaces.
These eigenfaces are used to find comparison weights for test image set
if the closest imagename is same as test image name then output is recognized
else it gives unknown person as result
"""
print(__doc__)


import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os
from numpy import linalg as LA

class Recognizer:
	
	
	def __init__(self, trainingDir):
		""" Initialize parameters , train the model and generate eigen faces over the training set """
		self.rowSize = 243
		self.colSize = 320
		noOfEigenFaces = 10
		f = os.listdir(trainingDir)
		self.imageNameSet = []	
		trainingSet = []
		fileCount = 0
		print "-------------------------"
		print 
		print "including images for training"
		print
		for file in f:
			img = Image.open(trainingDir + '/' + file).convert('RGBA')
			print file
			self.imageNameSet += [file]
			arr = np.array(img).tolist()
			trainingSet += [arr]
			fileCount += 1

			
		i = 0
		
		print
		print "converting samples to grayscale and calculating mean image ..."
		print

		j = 0
		self.mean = [0 for x in range(self.colSize * self.rowSize)]
		for t in trainingSet:
			arr = t
			
			f = 0
			print 'calculating for sample ', j
			j += 1
			sample = [0 for x in range(self.colSize * self.rowSize)]
		
			for r in xrange(0,self.rowSize):
				for c in xrange(0,self.colSize):

					pix = arr[r][c]
					v = self.grayscale(pix[0]+0.0,pix[1]+0.0,pix[2]+0.0)
					self.mean[f] += v
					sample[f] = v
					f += 1
					
				
			trainingSet[i] = sample
			i += 1
		
		noOfSamples = len(trainingSet)

		noOfFeatures = self.rowSize * self.colSize


		for i in xrange(0,noOfFeatures):
			self.mean[i] = self.mean[i] / noOfSamples

			
		for t in trainingSet:
			for i in xrange(0,noOfFeatures):
				t[i] = t[i] - self.mean[i]
		

	
		trainingNpSet = np.array(trainingSet)
		trainingNpSetT = np.transpose(trainingNpSet)
		

		print
		print "calculating eigen faces ..."
		print

		prod = np.dot(trainingNpSet,trainingNpSetT)
		
		w, v = LA.eig(prod)

		maxEig = []

		for s in xrange(0,noOfEigenFaces):
			max1 = 0.0
			i = 0
			m = 0
			for egw in w:
				if (egw > max1) and (i not in maxEig):
					max1 = egw
					m = i
				i+= 1
			maxEig += [m]		
		
		maxEigV = []
		for m in maxEig:
			maxEigV += [v[:,m].tolist()]	
		
		eigVecFinal = np.array(maxEigV)
		self.eigFaces = np.dot(eigVecFinal,trainingNpSet)
		
		i = 0

		print
		print "calculating weight vectors for training set ..."
		print

		self.trainingWeights = []
		for t in trainingNpSet:
			a = trainingNpSetT[:,i]
			weightVec = np.dot(self.eigFaces,a)
			self.trainingWeights += [weightVec]
			i += 1




	def grayscale(self, p1, p2, p3):
		""" convert given image to grayscale format """
		return (p1 + p2 + p3) / 3
	

	def recognize(self, testImagePath):
		""" given a test image the function returns the result of comparison and closest training image to the test image """
		print 
		print "recognizing image ",testImagePath, " ... "
		print

		img = Image.open(testImagePath).convert('RGBA')	
		test = np.array(img).tolist()
		
		testVec = [0 for x in range(self.colSize * self.rowSize)]
		j = 0
		for r in xrange(0,self.rowSize):
			for c in xrange(0,self.colSize):
				v = (test[r][c][0] + test[r][c][1] + test[r][c][2]) / 3
				testVec[j] = v
				j += 1
		testWeightVec = self.getWeightsVec(testVec)	
				
		minDis,minIdx = self.compare(testWeightVec)

		print "closest image name = ",self.imageNameSet[minIdx]
		print "minimum Distance = ",minDis
		
		tmp = self.imageNameSet[minIdx].split('.')
		tok = testImagePath.split('/')
		testImgName = tok[-1].split('.')[0]
		closestMatch = tmp[0]
		
		print
		if closestMatch == testImgName:
			print "recognized the person"
			print
			return True , closestMatch , testImgName
		else:
			print "unknown person"	
			print
			return False , closestMatch , testImgName




	def getWeightsVec(self, sampleVec):
		i = 0
		for j in self.mean:
			sampleVec[i] = sampleVec[i] - self.mean[i]
			i += 1
			
		s = np.array(sampleVec).transpose()
		weightVec = np.dot(self.eigFaces,s)
		return weightVec

	
	def compare(self, testWeightVec):
		""" use 1 nearest neighbour to find minimum 
		distance from a training sample and test image """

		minDis = 9999999999.0
		s = 0
		minIdx = 0
		for sample in self.trainingWeights:
			i = 0
			totalDist = 0.0
			for w in testWeightVec:
				totalDist += abs(testWeightVec[i] - sample[i])
				i += 1
			if totalDist < minDis:
				minDis = totalDist
				minIdx = s
			s += 1		
		return minDis , minIdx	



	def printMeanImage(self):
		""" produce mean image from the mean vector of training set """
		j = 0
		grey = np.zeros((self.rowSize,self.colSize))

		for x in xrange(0,self.rowSize):
			for y in xrange(0,self.colSize):
				grey[x][y] = self.mean[j]
				j += 1
		plt.imshow(grey,cmap = cm.Greys_r)
		plt.show()		


	def printEigenFaces(self, count):
		""" produce top k eigen faces where k is given input in count var """
		egFacesCopy = np.copy(self.eigFaces)

		for e in xrange(0,min(count,10)):
			grey = np.zeros((self.rowSize,self.colSize))
			egf = egFacesCopy[e,:]
			j = 0
			max1 = 0.0
			for x in xrange(0,self.rowSize):
				for y in xrange(0,self.colSize):
					if egf[j] < 0.0:
						egf[j] = 0.0

					if egf[j] > max1:
						max1 = egf[j]
					j += 1
						
			j = 0		
			for x in xrange(0,self.rowSize):
				for y in xrange(0,self.colSize):
					egf[j] = (egf[j] * 255) / max1
					j += 1
					
			j = 0		
			for x in xrange(0,self.rowSize):
				for y in xrange(0,self.colSize):
					grey[x][y] = egf[j]
					j += 1
				
			plt.imshow(grey,cmap = cm.Greys_r)
			plt.show()		