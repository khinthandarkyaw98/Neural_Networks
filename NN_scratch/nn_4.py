
"""
Dot product for single ouput NEURON

"""
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# the dot product of two vectors does not change
# the output no matter in which order they are muliplied by
output = np.dot(inputs, weights) + bias
print(f"output = {output}")