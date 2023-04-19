"""
Dot Product for muliple output NEURONS
"""
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]
            ]

# to be specified 3 output neurons
biases = [2, 3, 0.5]

# the order of dot product does matter
# in the case of vectors and matrices
##########################################
# so the following code will give out
# the shape of 
# (3,4) * (4,) = (3,) + (3,) = (3,)
# therefore, weights must come first in dot operation
outputs = np.dot(weights, inputs) + biases

print(outputs)