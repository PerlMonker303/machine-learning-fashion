import numpy as np
import matplotlib.pyplot as plt

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
    rows = []
    for i in range(len(Y[0])):  # For each training example
        el = int(Y[0][i])  # Storing the current element in variable 'el'
        row = [0] * n_y  # Creating the array full of zeros
        row[el] = 1  # Marking the current item with 1
        rows.append(row)  # Appending the row to the array of rows
    return np.array(rows).T  # Returning the "true label" array as a Numpy array

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
        pic_vector = X[:, rnd].reshape(28, 28)
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
