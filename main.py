# Library imports
import matplotlib.image as pimg
from PIL import Image
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

# My imports
from helper import *
from feature_scaling import *
from model_mini_batch import *

'''
Outline of the project
*OLD
1 - load the training data set (apply pre-processing)
2 - scale the features
3 - structure the CNN
4 - train the CNN
5 - predict the accuracy on the training data set
6 - load the test data set (apply pre-processing)
7 - predict the accuracy on the test data set
8 - add custom images of clothes
*NEW
1 - load the training data set(apply pre-processing)
2 - scale the features
3 - create placeholders for TensorFlow
4 - initialize the parameters
'''

'''1. Load the training data set (+ pre-processing)'''
m_train = 100  # No. of training examples
img_size = 28  # in pixels
n_x = img_size  # No. of features (img_size * img_size)
no_channels = 1  # Gray scale image
no_labels = 10  # No. of labels
[X_train, Y_train] = load_dataset(m_train, img_size, "train")

# Transforming the Y_train array to a "true label" array - 1 for Yes, 0 for No
Y_train = transform_y_true_label(Y_train, no_labels).T

'''DEBUG - visualising sizes of training data'''
print ("number of training examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

'''DEBUG - visualising some random images'''
#visualise_random_images(X_train, m_train, 5)

'''2. Scale the features'''
# This step is done in order to improve the learning process and have no huge
# gaps between values in our data sets (helps Optimization algorithms)
[X_train, lmbda, mu] = feature_scaling(X_train)

'''3. Create placeholders for TensorFlow'''
[X, Y] = create_placeholders(img_size, no_channels, no_labels)
#print ("X = " + str(X))
#print ("Y = " + str(Y))

'''4 . Initialize the parameters'''
parameters = initialize_parameters()

'''5. Forward propagation'''
Z3 = forward_propagation(X_train, parameters)








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
#parameters = model_mini_batch(X_train, Y_train, layers_dims, mini_batch_size, learning_rate, num_epochs, True, lambd_reg)

# CONTINUE HERE
