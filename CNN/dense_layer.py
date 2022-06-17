import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, inputs, gradients):
        self.weights_grad = np.dot(inputs.T, gradients)
        self.biases_grad = np.sum(gradients, axis=0, keepdims=True)
        return np.dot(gradients, self.weights.T)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad