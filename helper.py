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
    '''
    A = np.eye(C)
    B = Y.reshape(-1)
    D = A[B]
    E = D.T
    '''
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
    :return: [X_train, Y_train]
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

    return Z, cache_conv


def convolution_backward(dZ, cache):
    '''
    Backward propagation for convolution function
    :param dZ: gradient of the cost with respect to the output of the conv layer (Z)
    :param cache: cache of values needed for backprop
    :return: dA_prev - gradient of the cost w.r.t. to the previous layer A_prev (m,n_H_prev,n_W_prev,n_C_prev)
             dW - gradient of the cost w.r.t. the weights W (f,f,n_C_prev,n_C)
             db - gradient of the cost w.r.t. the biases b (1,1,1,n_C)
    '''
    # Retrieving parameters
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize the outputs
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_padding(A_prev, pad)
    dA_prev_pad = zero_padding(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find corners of current slice
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # Slice
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, : , c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]  # Unpadding

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

def test_convolution_backward():
    Z, cache_conv = test_convolution_forward()
    print("[TEST] Convolution backward")
    dA, dW, db = convolution_backward(Z, cache_conv)
    print("dA_mean = ", np.mean(dA))
    print("dW_mean = ", np.mean(dW))
    print("db_mean = ", np.mean(db))
    print("[TEST/] Convolution backward")


# Pooling functions
def pooling_forward(A_prev, hparameters, type="max"):
    '''
    Implements the forward pass of the pooling layer
    :param A_prev: input of shape (m,n_H_prev,n_W_prev,n_C_prev)
    :param hparameters: dictionary containing 'f' and 'stride'
    :param type: 'max' or 'average'
    :return: A - output of the pool layer (m,n_H,n_W,n_C)
             cache - used for backward pass - contains input and hparameters
    '''

    # Retreive different values
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']

    # Dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix with zeros
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Define corners
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Take a slice
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute pooling operation
                    if type == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif type == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache

def test_pooling_forward():
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {'stride': 2, 'f': 3}

    print("[TEST] Pooling forward")
    A, cache = pooling_forward(A_prev, hparameters, "max")
    print("type = max")
    print("A = ", A)
    A, cache = pooling_forward(A_prev, hparameters, "average")
    print("type = average")
    print("A = ", A)
    print("[TEST/] Pooling forward")


def create_mask(x, type):
    '''
    Creates a mask from an input matrix x to identify the max entry of x
    :param x: Array of shape (f, f)
    :param type: 'max' or 'average'
    :return: mask - array of shape (f, f) with True at the pos of the max/average entry
    '''
    mask = 0
    if type == 'max':
        mask = x == np.max(x)
    elif type == 'average':
        mask = x == np.average(x)
    return mask

def test_create_mask():
    print("[TEST] Mask creation")
    np.random.seed(1)
    x = np.random.randn(2, 3)
    mask = create_mask(x, 'max')
    print('x = ', x)
    print('max mask = ', mask)
    mask = create_mask(x, 'average')
    print('average mask = ', mask)
    print("[TEST/] Mask creation")

def distribute_value(dz, shape):
    '''
    Distributes the input value in the matrix of dimension shape
    :param dz: input scalar
    :param shape: (n_H, n_W) shape of output matrix for which we want to distribute value of dz
    :return: a - array of size (n_H, n_W) for which we distributed the value of dz
    '''
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)

    a = np.ones(shape) * average

    return a

def pooling_backward(dA, cache, type = 'max'):
    '''
    Implements the backward propagation of the pooling layer
    :param dA: gradient of cost w.r.t. the out of pooling layer (same shape as A)
    :param cache: cache output from forward pass of pooling layer (layer's input and hparameters)
    :param type: 'max' or 'average'
    :return: dA_prev - gradient of cost w.r.t. the input of the pooling layer (same shape as A_prev)
    '''

    # Retrieving some values
    (A_prev, hparameters) = cache
    stride = hparameters['stride']
    f = hparameters['f']
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize output
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]  # Select training example from A_prev
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find corners of current slice
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    if type == 'max':
                        # Slice
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Add mask
                        mask = create_mask(a_prev_slice, type)
                        # Multiply by mask
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif type == 'average':
                        da = dA[i, h, w, c]  # Get the value a from dA
                        shape = (f, f)  # Define shape of filter
                        # Distribute it to get the correct slice of dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev

def test_pooling_backward():
    print('[TEST] Pooling backward')
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {'stride': 1, 'f': 2}
    A, cache = pooling_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pooling_backward(dA, cache, 'max')
    print('type = max')
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])
    print()
    dA_prev = pooling_backward(dA, cache, 'average')
    print('type = average')
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print('[TEST/] Pooling backward')


def softmax(Z):
    # Softmax Function
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def relu(Z):
    # Rectified Linear Unit Function used for the traversal of the Neural Network
    return np.maximum(0, Z)


def logarithm(Z):
    # Logarithm function that does not allow 0 values (because log(0) = -infinity)
    constant = 0.000001
    Z = np.where(Z == 0.0, constant, Z)
    return np.log(Z)