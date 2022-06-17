import numpy as np


class ActivationLayer:
    def __init__(self, activation_function='relu'):
        if(activation_function == 'sigmoid'):
            self.activation_function = self.sigmoid
        elif(activation_function == 'tanh'):
            self.activation_function = self.tanh
        else:
            self.activation_function = self.relU


    def relU(self, X, derivative=False):
        if derivative:
            return 1.0 * (X > 0)
        return X * (X > 0)


    def sigmoid(self, X, derivative=False):
        if derivative:
            return X * (1 - X)
        return 1 / (1 + np.exp(-X))


    def tanh(self, X, derivative=False):
        if derivative:
            return 1 - X ** 2
        return np.tanh(X)


    def forward(self, X):
        self.input = X
        self.output = self.activation_function(X)
        return self.output


    def backward(self, dE_dY):
        return dE_dY * self.activation_function(self.x, True)