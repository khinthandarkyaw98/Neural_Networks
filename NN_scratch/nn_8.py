"""
    Create Objects
"""

# weights should be between -1 and 1
# if weights go above 1, 
# NN will get bigger and bigger

# to avoid NN producing '0' value
# set bias non-negative numbers

"""
The programmer has to know two things here.
1. the size of the input coming into the NN
2. the number of neurons included in the NN
"""
import numpy as np

np.random.seed(0)

# to generalize
# we will pass input X in batches
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # multplying by 0.10 will put the values of weights between -1 and 1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # here we do not need to transpose the weights
        # as the shape of weights is defined as (4 : the number of columns of inputs, output)
        self.output = np.dot(inputs, self.weights) + self.biases

# layer1
# n_inputs : the number of columns of X
# n_nuerons : as you like        
layer1 = Layer_Dense(4, 5)
# layer2
# n_inputs : the number of colums of layere1
# n_neurons : as you like
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(f"layer1.output = {layer1.output}")
layer2.forward(layer1.output)
print(f"layer2.output = {layer2.output}")
        
"""
layer1.output = [[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
 [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
 [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]]
layer2.output = [[ 0.148296   -0.08397602]
 [ 0.14100315 -0.01340469]
 [ 0.20124979 -0.07290616]]
 
 As seen above, 
 the number of rows of output of each layer = the number of rows of output of inputs
 the number of columns of output of each layer = the number of columns of layer_Dense
"""