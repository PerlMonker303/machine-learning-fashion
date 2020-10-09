# Library imports
import csv
import matplotlib.image as pimg
from PIL import Image
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

# My imports
from helper import *
from feature_scaling import *
from model_mini_batch import *

'''
Outline of the project
1 - load the training data set (apply pre-processing)
2 - scale the features
3 - structure the CNN
4 - train the CNN
5 - predict the accuracy on the training data set
6 - load the test data set (apply pre-processing)
7 - predict the accuracy on the test data set
8 - add custom images of clothes
'''

'''1. Load the training data set (+ pre-processing)'''
X_train_original = []  # Default Python array containing the features
Y_train_original = []  # Default Python array containing the outcomes
m_train = 10000  # No. of training examples
img_size = 28  # in pixels
n_x = img_size  # No. of features (img_size * img_size)
no_labels = 10  # No. of labels
index = 0
with open('./data/fashion-mnist_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',');  # Reading data
    for row in csv_reader:
        if index > 0:
            X_train_original.append(row[1:])
            Y_train_original.append(row[0:1]);
        index += 1
        if index > m_train:  # Reading only m_train examples
            break;

# Working with numpy arrays
X_train = np.array(X_train_original).T;  # n_x X m_train
Y_train = np.array(Y_train_original).T;  # 1   X m_train
X_train = X_train.astype('float64')  # Changing the dtypes to float64
Y_train = Y_train.astype('float64')

# Transforming the Y_train array to a "true label" array - 1 for Yes, 0 for No
Y_train = transform_y_true_label(Y_train, no_labels)

'''DEBUG - visualising some random images'''
visualise_random_images(X_train, m_train, 5)

'''2. Scale the features'''
# This step is done in order to improve the learning process and have no huge
# gaps between values in our data sets (helps Optimization algorithms)
[X_train, lmbda, mu] = feature_scaling(X_train)

'''3. Structure the Artificial Neural Network'''
n_x = 784  # No. of input units
n_h = []  # Array with the layers' dimensions
n_y = no_labels  # No. of output units
layers_dims = (n_x, n_h, n_y)  # Grouping the dimensions in a tuple

'''4. Training the Artificial Neural Network'''
learning_rate = 0.003
lambd_reg = 0.3  # The regularization factor
mini_batch_size = 32  # Setting the size of a batch (should be a power of 2)
num_epochs = 5  # Setting the no. of epochs
parameters = model_mini_batch(X_train, Y_train, layers_dims, mini_batch_size, learning_rate, num_epochs, True, lambd_reg)

# CONTINUE HERE
