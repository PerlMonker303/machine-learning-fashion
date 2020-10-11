import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf2
from tensorflow.python.framework import ops

'''
    Module with methods used for accomplishing different tasks which we must do more than once
'''


def transform_y_true_label(Y, n_y):
    '''
    Transforming the Y array into a "true label" array - 1 for Yes, 0 for No
    :param Y: original Y array
    :param n_y: number of labels
    :return: Y_modified
    '''
    '''
    rows = []
    for i in range(len(Y[0])):  # For each training example
        el = int(Y[0][i])  # Storing the current element in variable 'el'
        row = [0] * n_y  # Creating the array full of zeros
        row[el] = 1  # Marking the current item with 1
        rows.append(row)  # Appending the row to the array of rows
    return np.array(rows).T  # Returning the "true label" array as a Numpy array
    '''
    C = 10
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def visualise_random_images(X, m, no):
    '''
    Visualise a fixed amount of random images from a given data set
    :param X: Numpy array of images
    :param m: number of total images in the array
    :param no: number of pictures we want to visualise
    :return: None
    '''
    np.random.seed()  # initialise a random seed
    for i in range(no):
        rnd = np.random.randint(0, m)
        pic_vector = X[rnd]
        plt.imshow(pic_vector)
        plt.show()


def initialise_parameters_randomly(layers_dims):
    '''
    Randomly initialises the parameters (weights and biases) using He Initialisation
    :param layers_dims: tuple consisting of (n_x,n_h,n_y)
    :return: the randomly initialised parameters
    '''
    (n_x, n_h, n_y) = layers_dims
    parameters = dict()
    dimensions = [n_x] + n_h + [n_y]
    for l in range(1, len(dimensions)):  # Using He Initialization
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1]) * np.sqrt(2 / dimensions[l - 1])
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    return parameters


def load_dataset(m_train, img_size, typ):
    # Gray scale images
    X_train_original = []  # Default Python array containing the features
    Y_train_original = []  # Default Python array containing the outcomes
    index = 0
    with open('./data/fashion-mnist_'+typ+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',');  # Reading data
        for row in csv_reader:
            if index > 0:
                X_train_original.append(row[1:])
                Y_train_original.append(row[0:1]);
            index += 1
            if index > m_train:  # Reading only m_train examples
                break;

    X_train = np.array(X_train_original)
    Y_train = np.array(Y_train_original)
    X_train = X_train.astype('uint8')  # Changing the dtypes to float64
    Y_train = Y_train.astype('uint8')
    X_train = np.reshape(X_train, (X_train.shape[0], img_size, img_size, 1))
    Y_train = Y_train.reshape((1, Y_train.shape[0]))

    return [X_train, Y_train]


def create_placeholders(img_size, no_channels, no_labels):
    X = tf.placeholder(tf.float32, shape=(None, img_size, img_size, no_channels), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, no_labels), name='Y')
    return [X, Y]


def initialize_parameters():
    tf.set_random_seed(1)

    init = tf2.initializers.GlorotUniform()  # Xavier Initializer
    W1 = tf.Variable(init(shape=(4, 4, 3, 8)))
    W2 = tf.Variable(init(shape=(2, 2, 8, 16)))
    parameters = {"W1": W1, "W2": W2}
    return parameters

def forward_propagation(X, parameters):
    X = X.astype('float32')
    W1 = parameters['W1']
    W2 = parameters['W2']

    # First we apply the convolution operation
    # CU WEIGHT CRED CA E CEVA IN NEREGULA, CITESTE EROAREA
    tf2.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    #Z1 = tf2.nn.conv2(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # Then we apply the ReLU function
    #A1 = tf2.nn.relu(Z1)
    # Now we maxpool to diminish the size
    #P1 = tf2.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    return []