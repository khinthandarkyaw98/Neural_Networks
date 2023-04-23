"""
Instead of hardcoding in nn_18_implementing_loss1.py
we can use range(len())
"""
import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets  = [0, 1, 1]


print(f"len(softmax_outputs) = {len(softmax_outputs)}") #3
# range(3) -> 0, 1, 2

print(softmax_outputs[range(len(softmax_outputs)), class_targets])