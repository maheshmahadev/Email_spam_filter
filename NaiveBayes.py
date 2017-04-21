import csv
import random
import math


#loading the csv file stored in home forlder to read it line by line to create a dataset
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb")) #rb means ReadBinary mode
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]] #change the value to float from string
	return dataset




#function definition to split the dataset into training and testing
def splitDataset(dataset , splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy)) #choosing random rows from the dataset and appending them to the training set
		trainSet.append(copy.pop(index)) #copy is the testset
	return [trainSet, copy]


def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated): #[-1] indicates the last column in the dataset which is actually the class value
#check if there is a vector[0] or vector[1]. If not, create an empty value for separated of that value and append to it.
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated



def mean(numbers): #define the mean for each attrivute for a given class value such as a 0 or a 1.
	return sum(numbers)/float(len(numbers))



def stdev(numbers): #function to find standard deviation which is the square root of variance
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)



def summarize(dataset): #find mean and standard deviation for each attribute for each class. Do not consider the last column in the dataset since that column denotes the class value
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries



def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries



def calculateProbability(x, mean, stdev): #Gaussian density function
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent



def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities



def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel



def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions



def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



def main():
	filename = 'spambase.data.csv'
	splitRatio = 0.7
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)
 
import time
start_time = time.time()
main() 
print("%s seconds" % (time.time() - start_time))
