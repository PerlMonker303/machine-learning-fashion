import math

from helper import *

def model_cnn(X, Y, mini_batch_size, learning_rate, num_epochs, print_cost, lambd ):
    '''
   Models a convolutional neural network and trains the
   parameters using the Mini-Batch Gradient Descent Optimization algorithm
   :param X: input data of shape (n_x, no. of examples)
   :param Y: vector of elements from 0 to 10 (corresponding to each type of clothing)
   :param mini_batch_size: size of a mini-batch (1 = Stochastic G.D., m = Batch G.D.
   :param learning_rate: for Gradient Descent
   :param num_epochs: number of iterations for the optimization loop
   :param print_cost: if set to True the function will print the cost for each 100 iterations
   :param lambd: regularization factor
   :return: a dictionary containing the trained W1, W2, b1 and b2
   '''

    # First we define the architecture of the network. In order to do that we first need
    # to understand the tools we have in our shed.
    # There are three types of layers:
    # - Convolution (conv)
    # - Pooling (pool)
    # - Fully-Connected (fc)
    # Now there are many ways in which you can combine these layers, however there are some
    # well-known model which have been studied throughout history.
    # One such notable model is LeNet-5 developed in 1998 by Yann Lecun in order to
    # identify handwritten digits for zip code recognition in the postal service. This model
    # is considered by many to be the pioneering model which changed the way we see CNNs.
    # Foreword: we use max pooling instead of avg pooling as it is specified in the original paper.
    # STEPS:
    # input (28x28x1) =>
    # => (conv s=1, f=6 of size 5x5, p=0) 24x24x6 =>
    # => (pool max s=2, f=2, p=0) 12x12x6 =>
    # => (conv s=1, f=16 of size 5x5, p=0) 8x8x16 =>
    # => (pool max s=2, f=2, p=0) 4x4x16 (=256 nodes) =>
    # => (fc) 120x1x1 =>
    # => (fc) 84x1x1 =>
    # => output softmax layer (10x1x1)

    costs = []  # list used to keep track of the costs
    m = X.shape[0]  # no. of examples
    # Define sizes and types of layers
    n_x = (28, 28, 1)  # (n_w, n_h, n_c)
    n_y = (10, 1, 1)
    n_hdims = [(5, 5, 6), (5, 5, 16), (120, 1, 1), (84, 1, 1)]  # Dimensions
    n_h = [
        (6, 1, 0),  # (f, stride, pad)
        (16, 1, 0)
    ]
    '''
    ('conv', (1, 0, 6, 5, 5)),  # (conv, (s, p, f, size_f, size_f))
    ('maxpool', (2, 0, 2)),  # (maxpool, (s, p, f))
    ('conv', (1, 0, 16, 5, 5)),
    ('maxpool', (2, 0, 2)),
    ('fc', (120, 1, 1)),  # (fc, (n_w, n_h, n_c))
    ('fc', (84, 1, 1))
    '''

    dimensions = [n_x] + n_hdims + [n_y]

    # Randomly partition the batches
    mini_batches = list()
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    no_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, no_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k, mini_batch_size * (k + 1)]
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
    parameters = initialise_parameters_randomly(dimensions)

    # Start Optimization - Mini-Batch Gradient Descent
    activations = dict()  # Dictionary to hold the activation vectors A_0, A_1, ..., A_(L-1)
    cache = dict()  # Dictionary in which we cache the results of Z (we use them for the back propagation part)
    for i in range(0, num_epochs):
        for j in range(len(mini_batches)):
            mini_batch_X, mini_batch_Y = mini_batches[j]
            # Forward Propagation Step
            activations['A' + str(0)] = mini_batch_X
            # First we go through the Convolutional Layers (n_h)
            for k in range(0, len(n_h)):
                A_prev = activations['A' + str(k)]  # Take the previous activation layer
                W = parameters['W' + str(k)]  # Retrieve the weights from the parameters dictionary
                b = parameters['b' + str(k)]  # Retrieve the bias from the parameters dictionary
                f = n_h[k][0]
                stride = n_h[k][1]
                pad = n_h[k][2]
                hparameters = {'f': f, 'pad': pad}
                Z, _ = convolution_forward(A_prev, W, b, hparameters)
                cache['Z' + str(k + 1)] = Z

                hparameters = {'f': f, 'stride': stride}
                A, Z = pooling_forward(Z, hparameters, type="max")
                activations['A' + str(k + 1)] = A
                cache['Z' + str(k + 1)] = Z

            # Cost computation + Regularization
            A_last = A  # Taking the last activation vector
            cost = - 1 / m * np.sum(
                np.dot(mini_batch_Y.T, logarithm(A_last)) + np.dot((1 - mini_batch_Y).T, logarithm(1 - A_last)))
            regularization = 0  # Initialising the regularization
            for l in range(1, len(dimensions)):
                W = parameters['W' + str(l)]
                regularization += np.sum(np.square(W))

            cost += (lambd / m * regularization)
            costs.append(cost)

        print("Finished epoch {}".format(i))

    # Plot the cost graph at the end

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('# iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


'''
Some useful information:
- pooling === sub-sampling
- padding: used with convolving to prevent image shrinkage
- n x n matrix is convolved to dimensions n+2p-f+1 x n+2p-f+1 (p=padding, f=filter size)
- in practice we use two types of paddings:
- + Valid convolutions => no padding
- + Same convolutions => pad such that the output size stays the same as the input size
-   for this we use the following formula for the padding: p = (f-1)/2 (f is odd by convention) 
- stride: how many steps at a time the filter is moved
- convolving: n x n * f x f => floor((n + 2p - f) / s + 1) x floor((n + 2p - f) / s + 1)
- now for multi-channeled layers: n x n x n_c * f x f x n_c = n-f+1 x n-f+1 x n_c' where n_c is the no. of channels
- pooling: n x n x n_c => floor((n - f) / s + 1) x floor((n - f) / s + 1)
'''