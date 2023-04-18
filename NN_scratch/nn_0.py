# every neuron has a conncetion to each neuron in previous layer

# to get the desried output
# we have inputs
inputs = [1.2, 5.1, 2.1]

# every input has unquie weight assciated with
weights = [3.1, 2.1, 8.7]

# every unique neuron has a bias
bias = 3

# inputs * weights + bias
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print(f"output = {output}")
