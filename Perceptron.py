import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, num_inputs, iterations, learning_rate):
        self.iterations = iterations
        self.lr = learning_rate
        self.weights = np.zeros(num_inputs + 1)

    def predict(self, example):
        pred = np.dot(example, self.weights[1:]) + self.weights[0]
        return 1 if pred >= 0 else 0

    def train(self, examples, labels):
        for i in range(self.iterations):
            for example, label in zip(examples, labels):
                prediction = self.predict(example)
                for idx in range(len(self.weights) - 1):
                    self.weights[idx + 1] += self.lr * (label - prediction) * example[idx]
                self.weights[0] += self.lr * (label - prediction)

    def plot_predict(self, x1, y1):
        return self.predict([x1, y1])


if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y1 = df.iloc[0:100, 4].values
    y1 = np.where(y1 == 'Iris-setosa', 0, 1)

    X1 = df.iloc[0:100, [0, 2]].values

    P = Perceptron(2, 10, 0.01)
    P.train([[1, 1], [1, 0], [0, 1], [0, 0]], [1, 1, 1, 0])
