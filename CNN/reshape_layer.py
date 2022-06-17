import numpy as np


class ReshapeLayer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape


    def forward(self, input):
        self.input = input
        return np.reshape(input, self.output_shape)


    def backward(self, dE_dY, learning_rate):
        return np.reshape(dE_dY, self.input_shape)