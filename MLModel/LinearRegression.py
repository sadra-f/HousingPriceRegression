import numpy as np
from Visualize.plot import plot_on_process
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class LinearRegression:
    def __init__(self, learning_rate=0.003, num_iterations=100000, plot_loss=False):
        self.learning_rate = np.float64(learning_rate)
        self.num_iterations = num_iterations
        self.weights = None
        self.X = None
        self.y = None
        self.means = None
        self.normal_min = -3
        self.normal_max = 3
        self._processes = []
        self._loss_history = []
        self._hist_ready = True
        self._hist_path = "tmp/loss_history.txt"
        if not os.path.exists(os.getcwd() + "/tmp"):
            os.makedirs("tmp/")
        if plot_loss:
            np.savetxt(self._hist_path, [])
            tmp = plot_on_process(self._hist_path, "loss Function")
            self._processes.append(tmp)


    def train(self, X, y):
        self._loss_history = []
        self.org_X = X
        self.X = X
        self.y = y
        # self.y = self.normalize(self.y)
        # self.X = self.normalize(self.org_X)
        self.X_size = len(self.X)
        
        self.initialize_weights()
        for i in range(self.num_iterations):
            predictions = self._predict(self.X)
            if i % 1000 == 0:
                _loss = self.calc_loss(predictions, self.y)
                print(i, '->',_loss)
                self._loss_history.append(_loss)
                np.savetxt(self._hist_path, self._loss_history)
            self.adjust_weights(predictions)
        
        return self

    def adjust_weights(self, predictions):
        difference = np.array(predictions - self.y).reshape((self.X_size, 1))
        self.weights[0, 0] = self.weights[0, 0] - self.learning_rate * (np.sum(difference) / len(predictions))

        self.weights[0, 1:] = self.weights[0, 1:] - self.learning_rate * (np.sum(difference * self.X, axis=0) / self.X_size)

    def _predict(self, X):
        return np.sum(X * self.weights[0, 1:], axis=1) + self.weights[0, 0]

    def initialize_weights(self):
        self.weights = np.ones((1, len(self.X[0])+1))

    def normalize(self, vec, only_mixmax=False):
        """Initially performs Mean Normalization and then scales the normalized values to new range

        Args:
            X (np.array): dataset matrix as numpy array

        Returns:
            np.array : normalized X
        """
        if not only_mixmax:
            mean = np.mean(vec, axis=0)
            max = np.max(vec, axis=0)
            min = np.min(vec, axis=0)
            normalized_vec = (vec - mean) / (max - min)
        else:
            normalized_vec = vec
        # max = np.max(normalized_vec, axis=0)
        # min = np.min(normalized_vec, axis=0)
        # normalized_vec = self.normal_min + (((normalized_vec - min)*(self.normal_max - self.normal_min))/(max - min))
        return normalized_vec
    
    def denormalize(self, new_vec, old_vec):
        max_old = np.max(old_vec, axis=0)
        min_old = np.min(old_vec, axis=0)
        old_mean = np.mean(old_vec, axis=0)
        denormalized = new_vec * (max_old - min_old) + old_mean
        return denormalized

    def calc_loss(self, predictions, Y):
        diff = predictions - Y
        diff *= diff
        return  np.sum(diff) / (2 * len(predictions))

    def save_model(self, filename):
        np.savetxt(filename, self.weights)

    def load_model(self, filename):
        self.weights = np.loadtxt(filename)
        self.weights = self.weights.reshape((1, len(self.weights)))

    def predict(self, test_X):
        self.normalize(test_X)
        return self._predict(test_X)

    def __del__(self):
        for prcs in self._processes:
            prcs.join()