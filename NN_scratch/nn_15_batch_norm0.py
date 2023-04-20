import numpy as np

# use pip install nnfs to keep version same as that of nnfs book
import nnfs 

#np.random.seed(0)
# we do not need seed now
# let's use nnfs
nnfs.init()

# outputs in batches
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
print(f"exp_values = {exp_values}")

norm_base_test1 = np.sum(layer_outputs)
print(f"norm_base_test1 = {norm_base_test1}")
"""
norm_base_test1 = 18.172
"""



norm_base_test2 = np.sum(layer_outputs, axis = 1)
print(f"norm_base_test2 = {norm_base_test2}")
"""
norm_base_test2 = [8.395 7.29  2.487]
"""

norm_base_test3 = np.sum(layer_outputs, axis=1, keepdims=True)
print(f"norm_base_test3 = {norm_base_test3}")
"""
norm_base_test3 = [[8.395]
            [7.29 ]
            [2.487]]
"""

"""
Now, we know how to calculate norm_base by same row diff cols.
"""
norm_base = np.sum(exp_values, axis=1, keepdims=True)
norm_values = exp_values / norm_base
print(f"norm_values = {norm_values}")

"""
norm_values = [[8.95282664e-01 2.47083068e-02 8.00090293e-02]
                [9.99811129e-01 2.23163963e-05 1.66554348e-04]
                [5.13097164e-01 3.58333899e-01 1.28568936e-01]]
"""