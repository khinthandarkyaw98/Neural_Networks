import numpy as np
import nnfs 
import nn_12_spiral_data as sp

# to control random seed number
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        # we have to subtract to keep our program from being overflown 
        # this however will not have caused changes in our probabilites resulted
        exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True) 
        # we have to do the summation of values in the same row and keep its dimension
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)         
        self.output = probabilities
        
# samples = 100
# classes = 3        
X, y = sp.create_data(100, 3)

# 2 : we have X, y called 2 data points for 100 samples of 3 classes
dense1 = Layer_Dense(2, 3)   
# use ReLU for Hidden Layer
activation1 = Activation_ReLU()

# 3 : we want to classify 3 classes
dense2 = Layer_Dense(3, 3)
# use Softmax for final output layer
activation2 = Activation_Softmax()

# data from spiral_data
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# there will be (300, 3) answers for 3 classes
# let's print out the very first 5 rows for every columns included
print(activation2.output[:5, :])
#print(activation2.output[:5]) # same as above