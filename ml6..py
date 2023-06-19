print("\nNaive bayes classifier for concept learning problem")
import csv 
import random
import math
import operator
def safe_div(x,y):
    if y==0 :
        return 0
    return x/y

def loadcsv(filename):
    lines = csv.reader(open(filename))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset{i}=[float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset,splitratio):
    trainsize=int(len(dataset) * splitratio)
    trainset=[]
    copy=list(dataset)
    i=0
    while len(trainset) < trainsize:
        trainset.append(copy.pop(i))
    return [trainset,copy]

def sparatebyclass(dataset):
    separated={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if(vector[-1] not in separated):
            searated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return safe_div(sum(numbers),floa(len(numbers)))

def stdev(numbers):
    avg=mean(numbers)
    variance=safe_div(sum([pow(x-avg,2) for x in numbers]),float (len(numbers)-1))
    return math.sqrt(variance)

def summarize(dataset):
    summaries=[(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizebyclass(dataset):
    separated=separatebyclass(dataset)
    summarise[classvalue]=summarize(instances)
    return summaries

def calculateclassprobabilities(summaries,inputvectors):
    probabilities={}
    for classvalue,classsummaries in summaries.items():
        probabilities[classvalue]=1
        for i in range(len(classsummaries)):
           mean,stdev=classsummaries[i]
           x=inputvector[i]
           probabilities[classvalue]*=calculateprbability(x,mean,stdev)
    return probabilities

def predict(summaries,inputvector):
    probabilities=calculateclassprobabilities(summaries,inputvector)
    bestlabel,bestprob=None,-1
    for classvalue,probability in probabilities.item():
        if bestlable is None or probability > bestprob:
            bestprob=probability
            bestlable=classvalue
    return bestlable
        
        
def getaccuracy(testset,predictions):
    correct=0
    for i in range(len(testsets)):
        if testset[i][-1]==predictions[i]:
            correct+=1
    accuracy=safe_div(correct,float(len(testset)))*100.0
    return accuracy

def main():
    filename='Conceptlearning.csv'
    splitratio=0.9
    dataset=loadCsv(filename)
    trainingset,testset=splitdataset(dataset,splitratio)
    print("split {0} rows into ".format(len(dataset)))
    print("number of training data : "+(repr(len(trainingset))))
    print("number test data : "+(repe(len(testset))))
    print("\nThe values assumed for the concept learning attributes are\n")
    print("outlook=. sunny=>1 overcast=2 rain=3\n temparature=> hot=1 mild=2 cool=3 \n humidity=. high=1 normal=1\n wind=> weak=1 strong=1")
    print("TARGET CONCEPT:PLAY TENNIS=. YES=10 NO=5 ")
    print("\nTHE TRAINING SET ARE:")
    for x in trainingset:
        print(x)
    print("\nThe test data set are:")
    for x in testset:
        print(x)
    print('\n')
    summaries=summarizebyclass(trainingset)
    predictions=getpredictions(summaries,testset)
    actual=[]
    for i in range(len(testset)):
        vector=testset[i]
        actual.append(vector[-1])
    print("actual values : {0}%".format(actual))
    print('prediction:{0}%'.format(predictions))
    accuracy=getAccuracy(testset,predictions)
    print('Accuracy  :{0}%'.format(accuracy))
    
main()