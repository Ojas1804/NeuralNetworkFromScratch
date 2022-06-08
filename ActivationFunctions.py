import numpy as np

class ActivationFunction:
    def __init__(self, hidden_layer_activation_function, output_activation_function):
        self.hidden_layer_activation_function = self.sigmoid
        if(hidden_layer_activation_function == "tanh"):
            self.hidden_layer_activation_function = self.tanh
        elif(hidden_layer_activation_function == "relu"):
            self.hidden_layer_activation_function = self.relu

        self.output_activation_function = self.sigmoid
        if(output_activation_function == "softmax"):
            self.output_activation_function = output_activation_function


    def sigmoid(self, Z, derivative=False):
        ans = 1/(1 + np.exp(-Z))
        if derivative:
            return ans * (1 - ans)
        return ans


    def relu(self, Z, derivative=False):
        if derivative:
            return Z > 0
        return np.maximum(Z, 0)


    def tanh(self, Z, derivative=False):
        a = np.tanh(Z)
        if derivative:
            return 1 - a**2
        return a


    def softmax(self, Z, derivative=False):
        exp_scores = np.exp(Z - Z.max())
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if derivative:
            return probs * (1 - probs)
        return probs