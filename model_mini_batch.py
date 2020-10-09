import matplotlib.pyplot as plt
import math
import numpy as np
from helper import initialise_parameters_randomly

def model_mini_batch(X, Y, layers_dims, mini_batch_size, learning_rate, num_epochs, print_cost, lambd ):
    '''
    Implements a multi-layered artificial neural network and trains the
    parameters using the Mini-Batch Gradient Descent Optimization algorithm
    :param X: input data of shape (n_x, no. of examples)
    :param Y: vector of elements from 0 to 10 (corresponding to each type of clothing)
    :param layers_dims: tuple with the dimensions of the layers (n_x, n_h, n_y)
    :param mini_batch_size: size of a mini-batch (1 = Stochastic G.D., m = Batch G.D.
    :param learning_rate: for gradient descent
    :param num_epochs: number of iterations for the optimization loop
    :param print_cost: if set to True the function will print the cost for each 100 iterations
    :param lambd: regularization factor
    :return: a dictionary containing the trained W1, W2, b1 and b2
    '''

    costs = []  # list used to keep track of the costs
    m = X.shape[1]  # no. of examples
    (n_x, n_h, n_y) = layers_dims  # unboxing the layer dimensions

    # Randomly partition the batches
    mini_batches = list()
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    no_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, no_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k, mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k, mini_batch_size * (k + 1)]
        mini_batch = (mini_batch_X, mini_batch_Y)  # Pack it
        mini_batches.append(mini_batch)

    # Handling the case in which the last batch is not as big as the other batches
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,
                       mini_batch_size * (no_complete_minibatches - 1): mini_batch_size * no_complete_minibatches]
        mini_batch_Y = shuffled_Y[:,
                       mini_batch_size * (no_complete_minibatches - 1): mini_batch_size * no_complete_minibatches]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Initialise the parameters randomly
    parameters = initialise_parameters_randomly(layers_dims)

    # Start Optimization - Mini-Batch Gradient Descent
    activations = dict()  # Dictionary to hold the activation vectors A_0, A_1, ..., A_(L-1)
    cache = dict()  # Dictionary in which we cache the results of Z (we use them for the back propagation part)
    for i in range(0, num_epochs):
        pass
    # CONTINUE HERE
