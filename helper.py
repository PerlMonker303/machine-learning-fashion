import numpy as np
import matplotlib.pyplot as plt
import csv

'''
    Module with methods used for accomplishing different tasks applied more than once
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


# Convolution Functions
def zero_padding(X, pad):
    '''
    Adds zeros around the border of an image (used to avoid shrinking of the image after convolving)
    :param X: numpy array of shape (m,n_H,n_W,n_C) = batch of m images
    :param pad: integer, amount of padding around each image on vertical/horizontal dimensions
    :return: X_pad - padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    '''
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values=0)
    return X_pad

def test_zero_padding():
    print("[TEST] Zero Padding")
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_padding(x, 1) # apply a padding of 1
    print("x.shape =", x.shape)
    print("x_pad.shape =", x_pad.shape)
    print("x[1,1] =", x[1, 1])
    print("x_pad[1,1] =", x_pad[1, 1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])

    plt.show()

    print("[TEST/] Zero Padding")

def conv_single_step(a_slice_prev, W, b):
    '''
    Applies one filter defined by parameters W on a single slice of the output activation
    :param a_slice_prev: slice of input data of shape (f,f,n_C_prev)
    :param W: weight params contained in a window (f,f,n_C_prev)
    :param b: bias params contained in a window (1,1,1)
    :return: Z - scalar value, result of convolving the sliding window (W,b) on a slice x
     of the input data
    '''
    s = np.multiply(a_slice_prev, W)  # Multiply with the parameters W
    Z = np.sum(s)  # Sum over all entries
    Z = Z + float(b)  # Adding the bias
    return Z

def test_conv_single_step():
    print("[TEST] Convolve single step")
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)  # Initialize random parameters
    b = np.random.randn(1, 1, 1)
    Z = conv_single_step(a_slice_prev, W, b)
    print("Z=", Z)

    print("[TEST/] Convolve single step")

def convolve_window():
    pass

def convolution_forward(A_prev, W, b, hparameters):
    '''
    Implements the forward propagation for a convolution function
    :param A_prev: output activations of the previous layer (m,n_H_prev,n_W_prev,n_C_prev)
    :param W: weights
    :param b: biases
    :param hparameters: dictionary with 'stride' and 'pad'
    :return: Z - conv output (m,n_H,n_H,n_C)
             cache - cache of values needed for the convolution_backward()
    '''
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape  # Dimensions from A_prev's shape
    (f, f, n_C_prev, n_C) = W.shape  # Dimensions from W's shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    # We first compute the dimensions of the conv output volume
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Now the output volume Z may be initialized
    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_padding(A_prev, pad)  # Apply padding to previous layer

    for i in range(m):  # For each training example
        a_prev_pad = A_prev_pad[i]  # Select the ith training example's padded activation
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Finding the corners of the current slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Apply convolution on current slice
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[..., c], b[..., c])

    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hparameters)  # Cache the findings

    return Z, cache

def test_convolution_forward():
    print("[TEST] Convolution forward")
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2, "stride": 2}

    Z, cache_conv = convolution_forward(A_prev, W, b, hparameters)
    print("Z's mean = ", np.mean(Z))
    print("Z[3,2,1] = ", Z[3, 2, 1])
    print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])
    print("[TEST/] Convolution forward")


def convolution_backward():
    pass

# Pooling functions
def pooling_forward():
    pass

def create_mask():
    pass

def distribute_value():
    pass

def pooling_backward():
    pass