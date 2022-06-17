import numpy as np


class ANN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                        (self.hidden_nodes, self.output_nodes))
        self.biases_hidden_to_output = np.zeros((1, self.output_nodes))
        self.biases_input_to_hidden = np.zeros((1, self.hidden_nodes))

    def forward(self, inputs):
        self.hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)
        self.hidden_outputs = self.activation_function(self.hidden_inputs)

        self.final_inputs = np.dot(self.hidden_outputs, self.weights_hidden_to_output)
        self.final_outputs = self.activation_function(self.final_inputs)

        return self.final_outputs

    def backward(self, targets):
        self.final_outputs_error = targets - self.final_outputs
        self.final_outputs_delta = self.final_outputs_error * self.activation_function(self.final_inputs, True)

        self.hidden_outputs_error = np.dot(self.final_outputs_delta, self.weights_hidden_to_output.T)
        self.hidden_outputs_delta = self.hidden_outputs_error * self.activation_function(self.hidden_inputs, True)