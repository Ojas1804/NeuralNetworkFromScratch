import numpy as np
from math import floor

class PoolingLayer:
    def __init__(self, size, input_shape, stride, padding="valid"):
        self.size = size
        self.input_shape = input_shape
        self.stride = stride
        self.output_shape = np.floor(input_shape[0], self.get_length(input_shape[1]), 
                                     self.get_length(input_shape[2]))
        self.padding = padding
        
    def get_length(self, l):
        return floor((l - self.size + 1) / (2 * self.stride))

    def get_max(self, matr):
        return np.max(matr)

    def max_pool(self, input):
        output = np.zeros(self.output_shape)
        for k in range(self.input_shape[0]):
            for i in range(self.input_shape[1], self.stride):
                for j in range(self.input_shape[2], self.stride):
                    if i < self.output_shape[1] and j < self.output_shape[2]:
                        matr = input[k, i:i+self.stride, j:j+self.stride]
                        output[k][i / self.stride][j / self.stride] += np.max(matr)
        return output

    # def back_propagation(self, prev_gradient):
    #     dA = np.zeros(self.input_shape)
    #     for i in range(self.output_shape[0]):
    #         for j in range(self.output_shape)

