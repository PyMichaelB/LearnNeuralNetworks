import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)(1-sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return 0 if z<=0 else 1


class BasicNeuralNet:
    def __init__(self, num_inputs, num_in_hidden_layer, iterations, learning_rate, sample_size):
        self.iterations = iterations
        self.lr = learning_rate
        self.loss = []
        self.sample_size = sample_size

        # Initialise weights randomly
        self.weights1 = np.random.randn((num_inputs, num_in_hidden_layer))
        self.bias1 = np.random.randn(num_in_hidden_layer,)
        self.weights2 = np.random.randn(num_in_hidden_layer, 1)
        self.bias2 = np.random.randn(1,)

    def predict(self, example, label):
        self.z1 = np.dot(example, self.weights1) + self.bias1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return sigmoid(self.z2)

    def back(self, examples, y, y_dash):
        print("TO DO")

    def entropy_loss(self, y, y_dash):
        # We consider average entropy loss for a classification problem with 2 classes
        return (-1.0/self.sample_size) * np.sum(np.multiply(np.log(y_dash), y) + np.multiply((1-y), np.log(1-y_dash)))

