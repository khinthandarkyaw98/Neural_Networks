"""
   Batches
   
   Reasons of using batches : 
   
   1. To do parallel processing
        (the bigger the parallel processing is, the more opearitions we can run)
   2. when working on GPU for NN, it gives more faster efficiency
   3. Helps GENERALIZATION (otherwise, the model will overfit)
"""
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + biases

print(output)