import numpy as np
import matplotlib.pyplot as plt
from time import time
import DatasetHandler

class RBM:

    def __init__(self, max_epochs=10000, hidden_units=100, learning_rate=0.1, learning_rate_input_bias=0.1, learning_rate_hidden_bias=0.1, regularization=0.0002):
        self.max_epochs = max_epochs
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.learning_rate_input_bias = learning_rate_input_bias
        self.learning_rate_hidden_bias = learning_rate_hidden_bias
        self.regularization = regularization
        self.errors = []
        self.weights = np.array([])
        self.dream_data = 0
        self.hidden_representation = 0
        self.samples = 0
        self.num_of_dimensions = 0

    def sigmoid(self, values):
        return np.ones_like(values) / (np.ones_like(values) + np.exp(-values))
   
    def load_data(self, path='.\\Datasets\\train-images', pixel_threshold=75.0, batch_size=600):
        self.samples = DatasetHandler.read_idx(path)
        self.samples = DatasetHandler.apply_pixel_intensity_threshold(self.samples, pixel_threshold)
        self.samples = DatasetHandler.reshape_dataset(self.samples)
        num_of_samples, self.num_of_dimensions = self.samples.shape
        self.batches = DatasetHandler.divide_into_batches(self.samples, batch_size=batch_size)
        return self.batches

    def train(self):
        self.weights = np.random.randn(self.num_of_dimensions, self.hidden_units) * 0.1
        self.input_bias = np.zeros((1, self.num_of_dimensions))
        self.hidden_bias = np.zeros((1, self.hidden_units))

        for epoch in range(100000):
            print("Epoch {}".format(epoch))
            error = 0
            np.random.shuffle(self.batches)
            for data in self.batches:
                if not data.any():
                    continue
                # wake part
                wake_hidden_probabilities = self.sigmoid(np.dot(data, self.weights) + np.tile(self.hidden_bias, (len(data), 1)))
                wake_products = np.dot(data.T, wake_hidden_probabilities)
                wake_hidden_states = wake_hidden_probabilities > np.random.rand(len(data), self.hidden_units)
                wake_input_activation = sum(data)
                wake_hidden_activation = sum(wake_hidden_probabilities)

                # dream part
                dream_data = self.sigmoid(np.dot(wake_hidden_states, self.weights.T) + np.tile(self.input_bias, (len(data), 1)))
                dream_hidden_probabilities = self.sigmoid(np.dot(dream_data, self.weights) + np.tile(self.hidden_bias, (len(data), 1)))
                dream_products = np.dot(dream_data.T, dream_hidden_probabilities)
                dream_input_activation = sum(dream_data)
                dream_hidden_activation = sum(dream_hidden_probabilities)

                # weights update
                error += sum(sum(((data - dream_data)**2)))
                if error <= 1:
                    break
                self.errors.append(error)
                self.weights += self.learning_rate * ((wake_products - dream_products) / len(data) -  self.regularization * self.weights)
                self.input_bias += (self.learning_rate_input_bias / len(data)) * (wake_input_activation - dream_input_activation)
                self.hidden_bias += (self.learning_rate_hidden_bias / len(data)) * (wake_hidden_activation - dream_hidden_activation)
            print("Epoch {} completed, error: {}".format(epoch, error))

    def plot_error_curve(self):
        plt.figure()
        plt.plot(range(len(self.errors)), self.errors)
        plt.grid()
        plt.show()

    def generate_dream_data(self, data):
        data = np.array(data)
        if (len(data.shape) == 1):
            data = data.reshape(1, -1)
        wake_hidden_probabilities = self.sigmoid(np.dot(data, self.weights) + np.tile(self.hidden_bias, (len(data), 1)))
        wake_hidden_states = wake_hidden_probabilities > np.random.rand(len(data), self.hidden_units)
        self.dream_data = self.sigmoid(np.dot(wake_hidden_states, self.weights.T) + np.tile(self.input_bias, (len(data), 1)))
        return self.dream_data

    def generate_hidden_representation(self, data):
        self.dream_data = self.generate_dream_data(data)
        dream_hidden_probabilities = self.sigmoid(np.dot(self.dream_data, self.weights) + np.tile(self.hidden_bias, (len(data), 1)))
        self.hidden_representation = dream_hidden_probabilities
        return dream_hidden_probabilities

    def save_machine(self):
        date = time()
        np.savetxt('.\\RBMs\\weights_{}.csv'.format(date), self.weights, delimiter = ',')
        np.savetxt('.\\RBMs\\input_bias_{}.csv'.format(date), self.input_bias, delimiter = ',')
        np.savetxt('.\\RBMs\\hidden_bias_{}.csv'.format(date), self.hidden_bias, delimiter = ',')

    def load_machine(self, paths):
        self.weights = np.loadtxt(paths[0], delimiter=',')
        self.input_bias = np.loadtxt(paths[1], delimiter=',')
        self.hidden_bias = np.loadtxt(paths[2], delimiter=',')

    def save_dream_data(self):
        date = time()
        np.savetxt('.\\Datasets\\dream_data_{}.csv'.format(date), self.dream_data, delimiter = ',')
    
    def save_hidden_representation(self):
        date = time()
        np.savetxt('.\\Datasets\\hidden_representation_{}.csv'.format(date), self.hidden_representation, delimiter = ',')
    

'''
def standardize(X):
    means = X.mean(axis =1)
    stds = X.std(axis= 1, ddof=1)
    X= X - means[:, np.newaxis]
    X= X / stds[:, np.newaxis]
    return np.nan_to_num(X)
'''
if __name__ == "__main__":
    rbm = RBM()
    paths = ['.\\RBMs\\weights_1587597688.2092822.csv', '.\\RBMs\\input_bias_1587597688.2092822.csv', '.\\RBMs\\hidden_bias_1587597688.2092822.csv']
    rbm.load_machine(paths)
    rbm.load_data('.\\Datasets\\train-images')
    rbm.generate_hidden_representation(rbm.samples)
    rbm.save_hidden_representation()