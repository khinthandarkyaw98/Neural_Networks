"""
input transformed to vector : for matrix multplication, activation, etc
input multiplied by weights : to learn complex relationships between input and outputs
add bias to the multiplaction of input by weights : do want a solution go throgh the orign, e.g. we want some y-intercepts, better fit the data
activation function : we want to either repress or amplify the signal achieved  
cost function : difference between the predicted value from the model and the actual ground truth value
cost function vs parameters graph : we wanna minimize the cost value from the local minima all the way up to the global minima
learning rate : from local minima to global minma will be determined by the learning rate
"""
# activation function
# gradient of the activation function
# add bias units : to shift the data to the centroid : for better learning
# feed forward
# encoding the label
# perform model learning
# calculate the cost function
# gradient with respect to the weights

# between binary data and python data types
import struct 
import numpy as np
# visualization
import matplotlib.pyplot as plt
import os
from scipy.special import expit

def load_data():
    with open('Data/train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8)) # '>' for intel processor, 'II' for unsigned integer, # 8 bytes of data files
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('Data/train-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    with open('Data/t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8)) # '>' for intel processor, 'II' for unsigned integer, # 8 bytes of data files
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('Data/t10k-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    return train_images, train_labels, test_images, test_labels

def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(64):
        # we will print out number 7
        img = img_array[label_array==7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
    
# train_x, train_y, test_x, test_y = load_data()

# visualize_data(train_x, train_y)

# column vector representation: Dog [0, 1], Cat [1, 0]
def enc_one_hot(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0])) # col : each training example
    for i, val in enumerate(y):
        one_hot[val, i] = 1.0
    return one_hot

# y = np.array([4, 5, 9, 0])
# z = enc_one_hot(y)

# print(y)
# print()
# print(z)

def sigmoid(z):
    # return (1/(1 + np.exp(-z))) # this can blow up 
    return expit(z) # so we better use scipy's expit
    
def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    
"""
You will see 
x between -10 and 10
sigmoid gives 0 for really small negative values
and 1 for really large positive values
"""    
# visualize_sigmoid()

"""
    sigmoid(z) = 1 / (1 + e^-z)
           = (e^z / (e^z + 1))          [multiplying numerator and denominator by e^z]
sigmoid'(z) = d/dz (e^z / (e^z + 1))
             = (d/dz e^z * (e^z + 1) - e^z * d/dz (e^z + 1)) / (e^z + 1)^2  [quotient rule]
             = (e^z * (e^z + 1) - e^z * e^z) / (e^z + 1)^2                   [using the chain rule to find d/dz (e^z + 1)]
             = e^z / (e^z + 1)^2
             = sigmoid(z) * (1 - sigmoid(z))
"""
def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

"""
the model does not always provides the correct prediction
bc initally we randomize the parameters
so we need to penalize the prediction
there cost function comes in
from which the model learns its mistakes in terms of backward propagation
"""

"""
This is a cost function for binary classification 
    J(y, y_hat) = -1/m * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
where y is the true binary label (0 or 1), 
y_hat is the predicted probability (between 0 and 1), and 
m is the number of samples.
"""

def cal_cost(y_enc, output): # y_enc = ground_truth, output = output from the model
    t1 = -y_enc * np.log(output)
    t2 = (1-y_enc) * np.log(1-output)
    cost = np.sum(t1 - t2) # we have minus before the sum in the formula, so let's take minus in the code eq, so it will be the plus, minus means penality, so we will have penalities for 0 and penalities for 1
    return cost

def add_bias_unit(X, where):
    # where is just row or column
    if where == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1)) # same size but with an extra column
        X_new[:, 1:] = X # from col 1 to len(col) : our old array, so col 0 is just an a column all filled with ones
    if where == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X # row 1 to len(row): our old array, so row 0 is just an a row all filled with ones
    return X_new

"""
    n_features = the initial input which is 28 x 28, flattened into column vector = 784
    n_output = 10 bc of handwritten number images from 0 to 9 for classification
    
"""
def init_weights(n_features, n_hidden, n_output):
    # initially, weights are random numbers between -1 and 1
    w1 = np.random.uniform(-1.0, 1.0, size = n_hidden * (n_features + 1))
    # it comes out in the wrong dimensionality, so we need to reshape it
    w1 = w1.reshape(n_hidden, n_features + 1)
    w2 = np.random.uniform(-1.0, 1.0, size = n_hidden * (n_hidden + 1))
    w2 = w2.reshpae(n_hidden, n_hidden + 1)
    w3 = np.random.uniform(-1.0, 1.0, size = n_output * (n_hidden + 1))
    w3 = w3.reshape(n_output, n_hidden +1)
    return w1, w2, w3
    

