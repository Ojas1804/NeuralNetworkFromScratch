import numpy as np
import pickle

class ANN:

    def __init__(self, node_per_layer, iters=20, alpha=0.05, hidden_layers=2,
                 activation_function="sigmoid"):
        self.iters = iters
        self.alpha = alpha # learning rate
        self.hidden_layers = hidden_layers
        self.layers = hidden_layers + 2
        self.node_per_layer = node_per_layer  # list of nodes per layer
        self.activation_function = self.set_activation_function(activation_function)
        self.weights, self.bias = self.initialize_parameters()


    def initialize_parameters(self):
        weights = {}
        bias = {}
        for i in range(self.layers - 1):
            W = np.random.randn(self.node_per_layer[i], self.node_per_layer[i+1]) / np.sqrt(self.node_per_layer[i])
            weights[f"W{i}"] = W
            # print(f"weights[W{i}] = ", weights[f"W{i}"].shape)
            
            B = np.random.randn(1, self.node_per_layer[i + 1])
            bias[f"B{i}"] = B
            # print(f"bias[B{i}] = ", bias[f"B{i}"].shape)

        return weights, bias


    def set_activation_function(self, activation_function):
        act_func = self.sigmoid
        if(activation_function == "tanh"):
            act_func = self.tanh
        elif(activation_function == "relu"):
            act_func = self.relu
        return act_func


    # UTILITY FUNCTIONS
    def relu(self, Z, derivative=False):
        if derivative:
            return Z > 0
        return np.maximum(0, Z)


    def softmax(self, Z, derivative=False):
        exp_scores = np.exp(Z - Z.max())
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if derivative:
            return probs * (1 - probs)
        return probs


    def sigmoid(self, Z, derivative=False):
        ans = 1/(1 + np.exp(-Z))
        if derivative:
            return (np.exp(-Z))/((np.exp(-Z)+1)**2)
        return ans


    def tanh(self, Z, derivative=False):
        a = np.tanh(Z)
        if derivative:
            return 1 - a**2
        return a


    def error_calculation(self, out, y):
        return 1/(2 * 10) * np.sum((out - y)**2)


    def one_hot_encode(self, y):
        arr = [0] * 10
        arr[y[0]] = 1
        return np.array(arr)


    def calculate_accuracy(self, out, y):
        out = np.array(out)
        out = out.reshape(out.shape[0], 1)
        y_ = y.reshape(y.shape[0], 1)
        sum = 0
        for i in range(y.shape[0]):
            if(out[i] == y_[i]):
                sum = sum + 1
        return sum/ y_.shape[0] * 100


    # FORWARD PASS
    def forward_pass(self, x):
        # convert input to 2-d array using reshape
        x = x.values.reshape(1, self.node_per_layer[0])
        A = []
        Z = []

        for i in range(self.layers - 1):
            # print(x.shape, "     ", self.weights[f"W{i}"].shape, "      ", self.bias[f"B{i}"].shape)
            Z.append(x.dot(self.weights[f"W{i}"]) + self.bias[f"B{i}"])
            if(i == (len(self.node_per_layer) - 2)):
                A.append(self.softmax(Z[-1]))
            A.append(self.activation_function(Z[-1]))
            x = A[-1]

        return A, Z


    # BACK PROPAGATION
    def back_prop(self, x, y_, A, Z):
        y_ = self.one_hot_encode(y_).reshape(1, self.node_per_layer[-1])
        dB = []
        dW = []

        dK_last = 2 / 10 * (A[-1] - y_) * self.softmax(Z[-1], derivative=True)
        dK = dK_last

        for i in range(self.layers - 1):
            temp = self.layers - i
            if i > 0:
                dK = dK.dot(self.weights[f"W{temp - 1}"].T) * self.activation_function(Z[temp - 2], derivative=True)

            if(i == 0):
                dB.append(np.sum(dK, axis=0, keepdims=True))
            else:
                dB.insert(0, np.sum(dK, axis=0))

            if(i == self.layers - 2):
                x = x.values.reshape(1, 784)
                dW.insert(0, np.outer(dK.T, x.T))
            else:
                dW.insert(0, np.outer(dK.T, A[temp - 3].T))

        return dW, dB


    # UPDATE PARAMETERS
    def update_parameters(self, dW, dB):
        for i in range(self.layers - 1):
            # print(self.weights[f"W{i}"].shape, "    ", (dW[i].T).shape, "    ", self.bias[f"B{i}"].shape, "    ", dB[i].shape)
            self.weights[f"W{i}"] = self.weights[f"W{i}"] - (self.alpha * dW[i].T)
            self.bias[f"B{i}"] = self.bias[f"B{i}"] - (self.alpha * dB[i])


    # TRAIN
    def train_model(self, X, y):
        y = y.reshape(y.shape[1] * y.shape[0], 1)
        for i in range(self.iters):
            predictions = []
            print(f"ITERATION {i + 1} : ")
            for j in range(y.shape[0]):
                x = X.iloc[j]
                
                A, Z = self.forward_pass(x)
                prediction = np.argmax(A[-1])
                predictions.append(prediction)

                dW, dB = self.back_prop(x, y[j], A, Z)
                self.update_parameters(dW, dB)
            print("     Accuracy : ", self.calculate_accuracy(predictions, y), "%")
            predictions.clear()
            print("-" * 40)


    # TEST
    def test_model(self, X_test, y_test):
        predictions = []
        y_test = y_test.reshape(y_test.shape[1] * y_test.shape[0], 1)
        for i in range(y_test.shape[0]):
            A, Z = self.forward_pass(X_test.iloc[i])
            prediction = np.argmax(A[-1])
            predictions.append(prediction)
        print("        ACCURACY ON TEST DATASET:")
        print("Accuracy:", self.calculate_accuracy(predictions, y_test), "%")

    
    def save_weight_and_bias(self, name):
        file_weights = open(name, "wb")
        pickle.dump(self.params, file_weights)
        file_weights.close()

        file_bias = open(name, "wb")
        pickle.dump(self.bias, file_bias)
        file_bias.close()

    def get_weight_and_bias(self, name):
        file_weights = open(name, "rb")
        self.weights = pickle.load(file_weights)
        file_weights.close()

        file_bias = open(name, "rb")
        self.weights = pickle.load(file_bias)
        file_bias.close()
