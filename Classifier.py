from sklearn import svm
from time import time
from RBM import RBM
import numpy as np
import matplotlib.pyplot as plt
import struct
import DatasetHandler

class SVMClassifier:

    def __init__(self, dual=False, C=1.0, max_iter=5000):
        self.dataset = None
        self.labels = None
        self.classifier = svm.LinearSVC(dual=dual, C=C, max_iter=max_iter)
        self.predictions = None

    def train(self, paths):
        self.dataset = np.loadtxt(paths[0], delimiter=',')
        self.labels = DatasetHandler.read_idx(paths[1])
        self.classifier.fit(self.dataset, self.labels)

    def predict(self, data):
        self.predictions = self.classifier.predict(data)
        return self.predictions
    
    def save_classifier(self):
        date = time()
        np.savetxt('.\\Classifiers\\coef_{}.csv'.format(date), self.classifier.coef_, delimiter=',')
        np.savetxt('.\\Classifiers\\intercept_{}.csv'.format(date), self.classifier.intercept_, delimiter=',')
        np.savetxt('.\\Classifiers\\classes_{}.csv'.format(date), self.classifier.classes_, delimiter=',')
    
    def load_classifier(self, paths):
        self.classifier.coef_ = np.loadtxt(paths[0], delimiter=',')
        self.classifier.intercept_ = np.loadtxt(paths[1], delimiter=',')
        self.classifier.classes_ = np.loadtxt(paths[2], delimiter=',')

if __name__ == "__main__":
    classifier = SVMClassifier(max_iter=10000)
    paths = ['.\\Datasets\\hidden_representation_1587639525.9953616.csv', '.\\Datasets\\train-labels']
    classifier.train(paths)
    classifier.save_classifier()