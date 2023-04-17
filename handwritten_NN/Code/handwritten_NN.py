# between binary data and python data types
import struct 
import numpy as np
# visualization
import matplotlib.pyplot as plt
import os

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
    
train_x, train_y, test_x, test_y = load_data()

visualize_data(train_x, train_y)