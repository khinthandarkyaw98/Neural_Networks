import numpy as np
import nn_12_spiral_data as sp

np.random.seed(0)

X, y = sp.create_data(100, 3)

class Layer_Dense:
    # input, output(neurons)
    def __init__(self, n_inputs, n_neurons):
        # take the size of cols of n_inputs as rows and n_neurons(outputs) as columns for initial weight values
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # you can also change the values of biases to non-zeros
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # Transpose is not needed: as we took the col of inputs as rows of weights and did the dot product in order of inputs and weights
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
# 2 means that  3 classifications of 100 features have 2 data points        
layer1 = Layer_Dense(2, 5)
activation = Activation_ReLU()

layer1.forward(X)
print(f"layer1.output = {layer1.output}")
activation.forward(layer1.output)
print(f"activation.output = {activation.output}")        