from src.classes.dataProcessor import dataProcessor
from src.classes.kNN import kNN as kNN

data = dataProcessor()
learningData = data.processData('./data/iris.data.learning')
testingData = data.processData('./data/iris.data.test')
X_test = data.deleteLabels(testingData)

kNN = kNN(3,learningData)
unsetLabels = kNN.predict(X_test)
print("Finall score: ",kNN.score(testingData,unsetLabels))
print("Accuracy: ",(kNN.score(testingData,unsetLabels)/len(testingData))*100)
