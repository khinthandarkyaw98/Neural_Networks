"""
normalization
"""
import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

# E = 2.71828182846
E = math.e

# use numpy exponential
exp_values = np.exp(layer_outputs)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)
    
print(f"norm_valeus = {norm_values}")
# you will see that the sum of norm values does not exceed 1
print(f"sum(norm_values) = {sum(norm_values)}")
