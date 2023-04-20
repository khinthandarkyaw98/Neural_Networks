"""
    Add layers
"""
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]

# Hidden Layer 1
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


# Hidden Layer 2
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# output of Hidden Layer 1
# we neeed to tranpose the weights
# as the shape of weights is defined as (3 : the number of rows of input, output)
layer1_output = np.dot(inputs, np.array(weights).T) + biases
# output of Hidden Layer 2
# input of the layer2 = layer1_output
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(f"layer2_output = {layer2_output}")