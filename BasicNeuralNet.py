import numpy as np
import matplotlib.pyplot as plt

import csv
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)(1-sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(x):
    # For an array
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class BasicNeuralNet:
    def __init__(self, layers, iterations, learning_rate, x, y):
        self.iterations = iterations
        self.lr = learning_rate
        self.losses = []

        # Training data
        self.examples = x
        self.labels = y

        # Initialise weights randomly
        np.random.seed(1)
        self.weights1 = np.random.randn(layers[0], layers[1])
        self.bias1 = np.random.randn(layers[1],)
        self.weights2 = np.random.randn(layers[1], layers[2])
        self.bias2 = np.random.randn(layers[2],)

    def forward(self):
        self.z1 = np.dot(self.examples, self.weights1) + self.bias1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        y_dash = sigmoid(self.z2)
        return y_dash

    def predict(self, testing_examples):
        z1 = np.dot(testing_examples, self.weights1) + self.bias1
        a1 = relu(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        pred = sigmoid(z2)
        return np.round(pred)

    def back(self, y_dash):
        """
        'Base' is the derivative of the loss w.r.t. z2 - all the other derivatives come from this so its useful
        MSE
        base = -(self.labels - y_dash) * y_dash * (1 - y_dash)

        LOG
        base = y_dash - self.labels
        """
        base = y_dash - self.labels
        dl_dw2 = np.dot(self.a1.T, base)
        dl_db2 = np.sum(base, axis=0)

        dl_dw1 = np.dot(self.examples.T, np.dot(base, self.weights2.T) * relu_prime(self.z1))
        dl_db1 = np.sum(np.dot(base, self.weights2.T) * relu_prime(self.z1), axis=0)

        self.weights1 -= self.lr * dl_dw1
        self.weights2 -= self.lr * dl_dw2
        self.bias2 -= self.lr * dl_db2
        self.bias1 -= self.lr * dl_db1

    def train(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for iteration in range(self.iterations):
            y_dash = self.forward()
            self.back(y_dash)
            self.losses.append(self.calculate_loss(self.labels, y_dash))

        ax.plot(self.losses)
        plt.show()

    def calculate_loss(self, y, y_dash):
        return  -1/len(y) * (np.sum(np.multiply(np.log(y_dash), self.labels) + np.multiply((1 - self.labels), np.log(1 - y_dash))))

    def accuracy(self, y, y_dash):
        return sum(y == y_dash) / len(y) * 100

