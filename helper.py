import numpy as np
import matplotlib.pyplot as plt
import csv

'''
    Module with methods used for accomplishing different tasks which we must do more than once
'''


def transform_y_true_label(Y, n_y):
    '''
    Transforming the Y array into a "true label" array - 1 for Yes, 0 for No
    :param Y: original Y array of the form [[... values ...]]
    :param n_y: number of labels
    :return: Y_modified
    '''

    ''' OLD VERSION
    rows = []
    for i in range(len(Y[0])):  # For each training example
        el = int(Y[0][i])  # Storing the current element in variable 'el'
        row = [0] * n_y  # Creating the array full of zeros
        row[el] = 1  # Marking the current item with 1
        rows.append(row)  # Appending the row to the array of rows
    return np.array(rows).T  # Returning the "true label" array as a Numpy array
    '''
    C = 10 # number of classes
    # [DEBUG] - to understand how it is done
    A = np.eye(C)
    B = Y.reshape(-1)
    D = A[B]
    E = D.T
    Y = np.eye(C)[Y.reshape(-1)].T  # eye gives an array of 1s with
    return Y



def visualise_random_images(X, m, no):
    '''
    Visualise a specific amount of random images from a given data set
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
        print(X[rnd])


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


def load_dataset(m, img_size, typ):
    '''
    Loads m values from the specified dataset
    :param m: how many examples to load
    :param img_size: size of image
    :param typ: "train" or "train"
    :return:
    '''
    # Gray scale images
    X_train_original = []  # Default Python array containing the features
    Y_train_original = []  # Default Python array containing the outcomes
    index = 0
    with open('./data/fashion-mnist_'+typ+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')  # Reading data
        for row in csv_reader:
            if index > 0:
                X_train_original.append(row[1:])
                Y_train_original.append(row[0:1])
            index += 1
            if index > m:  # Reading only m examples
                break

    # Creating numpy arrays
    X_train = np.array(X_train_original)
    Y_train = np.array(Y_train_original)
    X_train = X_train.astype('uint8')  # Changing the dtypes to uint8
    Y_train = Y_train.astype('uint8')
    X_train = np.reshape(X_train, (X_train.shape[0], img_size, img_size, 1))
    print(X_train)
    Y_train = Y_train.reshape((1, Y_train.shape[0]))

    return [X_train, Y_train]