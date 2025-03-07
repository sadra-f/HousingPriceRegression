import numpy as np
from Visualize.plot import plot_on_process
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pickle

class LinearRegression:
    def __init__(self, learning_rate=0.003, num_iterations=10000, plot_loss=False):
        self.learning_rate = np.float64(learning_rate)
        self.num_iterations = num_iterations

        self.X = None
        self.Y = None

        self.weights = None
        self.x_train_mean = None
        self.x_train_standard_deviation = None
        self.y_train_mean = None
        self.y_train_standard_deviation = None

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

    def train(self, X, Y):
        """train the model using the input training data

        Args:
            X (_type_): Train Dataset Features
            Y (_type_): Train Dataset Target

        Returns:
            LinearRegression: returns self
        """
        self._loss_history = []
        self.org_X = X
        self.org_Y = Y
        self._calc_mean_std(X, Y)
        self.X, self.Y = self._standardize(X, Y)

        self.X_size = len(self.X)
        
        self.initialize_weights()
        for i in range(self.num_iterations):
            predictions = self._predict(self.X)
            if i % 100 == 0:
                self._log_loss(i, predictions)
            self.adjust_weights(predictions)
        
        return self
    
    def _log_loss(self, iter, predictions):
        _loss = self.calc_loss(predictions, self.Y)
        print(iter, '->',_loss)
        self._loss_history.append(_loss)
        np.savetxt(self._hist_path, self._loss_history)
    
    def adjust_weights(self, predictions):
        """performs the 'learning' of the weights

        Args:
            predictions (np.ndarray): the predictions for current weights
        """
        difference = np.array(predictions - self.Y).reshape((self.X_size, 1))
        self.weights[0, 0] = self.weights[0, 0] - self.learning_rate * (np.sum(difference) / len(predictions))

        self.weights[0, 1:] = self.weights[0, 1:] - self.learning_rate * (np.sum(difference * self.X, axis=0) / self.X_size)

    def _predict(self, X):
        return np.sum(X * self.weights[0, 1:], axis=1) + self.weights[0, 0]

    def initialize_weights(self):
        self.weights = np.ones((1, len(self.X[0])+1))

    def _calc_mean_std(self, X, Y=None):
        """calculates the mean and standard deviation for each column of X and Y.

        Args:
            X (_type_): Train Dataset Features
            Y (_type_, optional): Train Dataset Target
        """
        self.x_train_mean = np.mean(X, axis=0)
        self.x_train_standard_deviation = np.std(X, axis=0)
        if Y is not None:
            self.y_train_mean = np.mean(Y)
            self.y_train_standard_deviation = np.std(Y)

    def _standardize(self, X, Y=None):
        normalized_X = (X - self.x_train_mean) / self.x_train_standard_deviation
        if Y is not None:
            normalized_Y = (Y - self.y_train_mean) / self.y_train_standard_deviation
            return normalized_X, normalized_Y
        
        return normalized_X
          
    def _destandardize(self, value, mean, std):
        return (value * std) + mean

    def calc_loss(self, predictions, Y):
        """Calculates the loss for predictions based on the original Y values

        Args:
            predictions (_type_): Predictions made by the model for Y/target
            Y (_type_): Original Y/target values

        Returns:
            _type_: The loss 
        """
        diff = predictions - Y
        diff *= diff
        return  np.sum(diff) / (2 * len(predictions))

    def save_model(self, filename):
        """Saves the model data in the specified file path

        Args:
            filename (path/str): Path to target file
        """
        model = {
            "xtm" : self.x_train_mean,
            "xtsd" : self.x_train_standard_deviation,
            "ytm" : self.y_train_mean,
            "ytsd" : self.y_train_standard_deviation,
            "w" : self.weights,
        }
        with open(f'{filename}', 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        """Loads pretrained model from file

        Args:
            filename (path/str): Path to target file
        """
        with open(f'{filename}', 'rb') as handle:
            model = pickle.load(handle)

        self.x_train_mean = model["xtm"]
        self.x_train_standard_deviation = model["xtsd"]
        self.y_train_mean = model["ytm"]
        self.y_train_standard_deviation = model["ytsd"]
        self.weights = model["w"][0].reshape((1, len(model["w"][0])))

    def predict(self, test_X):
        """Makes predictions for test_X values and returns the predictions.

        Args:
            test_X (ndarray): Values the predictions for which are required

        Returns:
            ndarray : predictions for test_X made by the regression model
        """
        test_X = self._standardize(test_X)
        predictions =  self._predict(test_X)
        return self._destandardize(predictions, self.y_train_mean, self.y_train_standard_deviation)
        
    def __del__(self):
        for prcs in self._processes:
            prcs.join()