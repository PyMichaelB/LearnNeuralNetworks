import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def activate(x):
    # Use below for Heaviside activation
    # return np.where(x>=0.0, 1, 0)

    # Use below for sigmoidal activation
    return 1/(1+np.exp(-x))


class Perceptron:
    def __init__(self, size, xs, ys):
        self.n = size
        self.weights = {}
        self.bias = 0
        self.xs = xs
        self.ys = ys

        self.lr = 0.01

    def produce_initial_params(self):
        for idx in range(self.n):
            self.weights['W' + str(idx)] = 0
            self.bias = 0

    def feed_forward(self, x_vec):
        # Just produces the predicted value for an example x_vec
        pred = 0
        for i in range(len(x_vec)):
            pred += self.weights['W' + str(i)] * x_vec[i] + self.bias
        return activate(pred)

    def calculate_loss(self):
        loss = 0
        for k in range(len(self.xs)):
            # sum of squares of difference between y and y'
            y_dash = self.feed_forward(self.xs[k])
            loss += (self.ys[k] - y_dash)*(self.ys[k] - y_dash)
        return loss

    def back_prop(self, i):
        # Work out changes in Wi and bi required
        dwi = 0
        dbi = 0

        for k in range(len(self.xs)):
            # go over every example, computing derivative of the loss
            y_dash = self.feed_forward(self.xs[k])
            delta = self.lr * (self.ys[k] - y_dash)
            dwi += self.xs[k][i] * delta
            dbi += delta

        return dwi, dbi

    def train(self):
        self.produce_initial_params()
        for iter in range(6):
            for i in range(self.n):
                # going over each feature and then updating the associated weight and bias
                self.weights['W' + str(i)] += self.back_prop(i)[0]
                self.bias += self.back_prop(i)[1]
        return


if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:150, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X = df.iloc[0:100, [0, 2]].values

    P = Perceptron(2, X, y)
    P.produce_initial_params()
    P.train()

    y_lin = np.linspace(0, 6, 20)
    x_lin = np.linspace(4, 8, 20)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = P.feed_forward([X, Y])
    ax.plot_surface(X, Y, Z)

    # Plot a 3D surface
    plt.show()

