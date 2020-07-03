import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def activate(x, b):
    # Use below for Heaviside activation
    return np.where(x >= -1*b, 1, 0)


class Perceptron:
    def __init__(self, size, xs, ys):
        self.n = size
        self.weights = {}
        self.xs = xs
        self.ys = ys
        self.lr = 0.01

    def produce_initial_params(self):
        # Setting all the weights to zero and the bias to zero
        self.weights['b'] = 0

        for idx in range(self.n):
            self.weights['W' + str(idx + 1)] = 0

        return self.weights

    def feed_forward(self, x_vec):
        # Just produces the predicted output for a given example x_vec
        pred = 0
        for idx in range(self.n + 1):
            if idx == 0:
                pred += self.weights['b']
            else:
                pred += self.weights['W' + str(idx)] * x_vec[idx - 1]

        return activate(pred, self.weights['b'])

    def calculate_loss(self):
        loss = 0
        for k in range(len(self.xs)):
            # Sum of squares of difference between y and y'
            y_dash = self.feed_forward(self.xs[k])
            loss += (self.ys[k] - y_dash)*(self.ys[k] - y_dash)
        return loss

    def back_prop(self):
        # Work out changes in Wi and bi required
        for idx in range(self.n + 1):
            change = 0
            for k in range(len(self.xs)):
                # Go over every example, add appropriate amounts
                y_dash = self.feed_forward(self.xs[k])
                delta = self.lr * (self.ys[k] - y_dash)

                if idx == 0:
                    change += delta
                else:
                    change += delta * self.xs[k][idx - 1]

            if idx == 0:
                self.weights['b'] += change
            else:
                self.weights['W' + str(idx)] += change

    def train(self):
        self.produce_initial_params()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for iteration in range(100):
            self.back_prop()
            ax.scatter(iteration, self.calculate_loss(), color='b')

        ax.set_title('Loss vs iteration number')


if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y1 = df.iloc[0:100, 4].values
    y1 = np.where(y1 == 'Iris-setosa', 0, 1)

    X1 = df.iloc[0:100, [0, 2]].values

    P = Perceptron(2, X1, y1)
    P.train()

    y_lin = np.linspace(0, 6, 30)
    x_lin = np.linspace(4, 8, 30)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = P.feed_forward([X, Y])
    ax.plot_surface(X, Y, Z)
    ax.set_title('Prediction plot')

    # Plot a 3D surface
    plt.show()

