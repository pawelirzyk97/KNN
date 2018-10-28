from src.classes.dataProcessor import getEuclideanDistance, lookForLabel
import operator

class kNN:

    def __init__(self, k, learningSet):
        self.k = k
        self.learningSet = learningSet

