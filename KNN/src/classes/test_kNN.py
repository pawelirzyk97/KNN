import unittest
from src.classes.dataProcessor import dataProcessor
from src.classes.kNN import kNN

data = dataProcessor()
testingData = data.processData('./data/iris.data.test')
learningData = data.processData('./data/iris.data.learning')

