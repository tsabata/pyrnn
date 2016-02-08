__author__ = 'sabata tomas'
import numpy as np


class RNN:
    def __init__(self,input_dim, hidden_dim, output_dim, alpha=0.1):
        self.alpha = alpha
        self.synapse_0 = 2*np.random.random((input_dim. hidden_dim)) - 1
        self.synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1
        self.synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

        self.synapse_0_update = np.zeros_like(self.synapse_0)
        self.synapse_1_update = np.zeros_like(self.synapse_1)
        self.synapse_h_update = np.zeros_like(self.synapse_h)
