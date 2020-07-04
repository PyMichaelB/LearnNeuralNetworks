import csv
import Perceptron
"""
A simple class to put fund data into the Perceptron to see whether it made a profit this year or not
"""


class Funds:
    def __init__(self, to_predict_example):
        self.fund_examples = []
        self.fund_labels = []
        self.example = to_predict_example

        with open('perceptron_fund_data.csv') as fund_data:
            my_reader = csv.reader(fund_data, delimiter=',')
            next(my_reader)
            for row in my_reader:

                self.fund_examples.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
                self.fund_labels.append(int(row[7]))

    def make_prediction(self):
        P = Perceptron.Perceptron(5, 100, 0.01)
        P.train(self.fund_examples, self.fund_labels)
        return P.predict(self.example)


if __name__ == "__main__":
    funds = Funds([491.92, 3.0, 52.0, 28.66, 27.89])
    print(funds.make_prediction())