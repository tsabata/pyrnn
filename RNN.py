__author__ = 'sabata tomas'
import copy
import numpy as np



# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.1, seed=None):
        """
        Initializes RNN.
        :param input_dim: Dimension of input layer
        :param hidden_dim: Dimension of hidden layer
        :param output_dim: Dimension of output layer
        :param alpha: Learning rate
        """
        # TODO: Check constraints
        self.alpha = alpha
        if seed:
            np.random.seed(seed)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
        self.synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
        self.synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

        self.synapse_0_update = np.zeros_like(self.synapse_0)
        self.synapse_1_update = np.zeros_like(self.synapse_1)
        self.synapse_h_update = np.zeros_like(self.synapse_h)

    def train(self, X):
        """
        Trains RNN
        For usage details check examples.
        :param X: Dataset - list of instances inherited from ProblemObject
        """
        # TODO: check if sizes coresspond with dimensions of layers
        # TODO: progress visualisation
        X = np.array(X)
        if not isinstance(X[0], ProblemObject):
            raise ValueError("X is not list of objects type ProblemObject. Inherit class ProblemObject")
        for index in range(X.shape[0]):
            problem_object = X[index]
            layer_2_deltas = list()
            layer_1_values = list()
            layer_1_values.append(np.zeros(self.hidden_dim))
            i = 0
            for seq_item in problem_object.sequence_iterator():
                input = np.array(seq_item)
                layer_1 = sigmoid(np.dot(input, self.synapse_0) + np.dot(layer_1_values[-1], self.synapse_h))
                layer2 = sigmoid(np.dot(layer_1, self.synapse_1))
                problem_object.predicted(layer2, i)
                layer_2_error = problem_object.error(i)
                layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer2))
                layer_1_values.append(copy.deepcopy(layer_1))
                i += 1
            future_layer_1_delta = np.zeros(self.hidden_dim)

            layer_pos = 0
            for seq_item in reversed(problem_object.sequence_iterator()):
                input = np.array(seq_item)
                layer_1 = layer_1_values[-layer_pos - 1]
                prev_layer_1 = layer_1_values[-layer_pos - 2]
                # error at output layer
                layer_2_delta = layer_2_deltas[-layer_pos - 1]
                # error at hidden layer
                layer_1_delta = (future_layer_1_delta.dot(self.synapse_h.T) + layer_2_delta.dot(
                    self.synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

                # let's update all our weights so we can try again
                self.synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                self.synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                self.synapse_0_update += input.T.dot(layer_1_delta)

                future_layer_1_delta = layer_1_delta
                layer_pos += 1

            # update weights
            self.synapse_0 += self.synapse_0_update * self.alpha
            self.synapse_1 += self.synapse_1_update * self.alpha
            self.synapse_h += self.synapse_h_update * self.alpha
            self.synapse_0_update *= 0
            self.synapse_1_update *= 0
            self.synapse_h_update *= 0

    def predict(self, problem_object):
        """
        Use learned RNN to predict problem object
        :param problem_object: Problem object to predict
        """
        if not isinstance(problem_object, ProblemObject):
            raise ValueError("X is not list of objects type ProblemObject. Inherit class ProblemObject")
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.hidden_dim))
        i = 0
        for seq_item in problem_object.sequence_iterator():
            input = np.array(seq_item)
            layer_1 = sigmoid(np.dot(input, self.synapse_0) + np.dot(layer_1_values[-1], self.synapse_h))
            layer2 = sigmoid(np.dot(layer_1, self.synapse_1))
            problem_object.predicted(layer2, i)
            layer_1_values.append(copy.deepcopy(layer_1))
            i += 1


class ProblemObject:
    def sequence_iterator(self):
        raise NotImplementedError("Method sequence_iterator is not implemented.\n"
                                  "Method return iterator over sequence")

    def error(self, index):
        raise NotImplementedError("Method error is not implemented.\n"
                                  "Method return error based on result (real - predicted)\n")

    def predicted(self, result, index):
        raise NotImplementedError("Method predicted is not implemented.\n"
                                  "Method fill predicted value of patt of sequence")

    def set_output(self, output):
        raise NotImplementedError("Method set_output is not implemented.\n"
                                  "Method save output vector for sequence")
