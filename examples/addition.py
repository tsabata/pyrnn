__author__ = 'sabata tomas'
import numpy as np
from RNN import RNN, ProblemObject


class BinaryAddition(ProblemObject):
    def __init__(self, a, b):
        self.iter_pos = 0
        self.output = None
        self.a = a
        self.b = b
        self.pred = []
        self.d = []

    def to_int(self, list):
        out = 0
        for index, x in enumerate(reversed(list)):
            out += x * pow(2, index)
        return out

    def sequence_iterator(self):
        # adding 2 binary numbers is computed from right to left
        reversed = []
        for i in range(len(self.a)):
            reversed.append([[self.a[-i - 1], self.b[-i - 1]]])
        return reversed

    def predicted(self, predicted, index):
        self.pred.append(predicted[0][0])
        self.d.insert(0, int(np.round(predicted[0][0])))

    def error(self, index):
        err = self.output[-index - 1] - self.pred[index]
        return err

    def set_output(self, output):
        self.output = output

    def print(self):
        print("Pred:" + str(np.array(self.d)))
        print("True:" + str(self.output))
        print(str(self.to_int(self.a)) + " + " + str(self.to_int(self.b)) + " = " + str(self.to_int(self.d)))
        print("------------")


# preparing dataset
size = 8
int2binary = {}
largest_number = pow(2, size)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

X = []
for j in range(10000):
    a_int = np.random.randint(largest_number / 2)  # int version
    a = int2binary[a_int]  # binary encoding
    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding
    c_int = a_int + b_int  # true answer
    c = int2binary[c_int]
    problem = BinaryAddition(a, b)
    problem.set_output(c)
    X.append(problem)

rnn = RNN(2, 16, 1)
rnn.train(X)

a = int2binary[np.random.randint(largest_number / 2)]  # binary encoding
b = int2binary[np.random.randint(largest_number / 2)]  # binary encoding
problem = BinaryAddition(a, b)

rnn.predict(problem)

problem.print()