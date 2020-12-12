def model_cnn(X, Y, layers_dims, mini_batch_size, learning_rate, num_epochs, print_cost, lambd ):
    '''
   Models a convolutional neural network and trains the
   parameters using the Mini-Batch Gradient Descent Optimization algorithm
   :param X: input data of shape (n_x, no. of examples)
   :param Y: vector of elements from 0 to 10 (corresponding to each type of clothing)
   :param layers_dims: tuple with the dimensions of the input/output layers (n_x, n_y)
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
    # One such notable model is LeNet-5 developed in 1998 by Yann Lecun in order tp
    # identify handwritten digits for zip code recognition in the postal service. This model
    # is considered by many to be the pioneering model which changed the way we see CNNs.
    # Foreword: we use max pooling instead of avg pooling as it is specified in the original paper.
    # STEPS:
    # input (28x28x1) =>
    # => (conv s=1, f=6 of size 5x5, p=0) 24x24x6 =>
    # => (pool max s=2, f=2, p=0) ... => ...
    pass




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
- n x n * f x f => floor((n + 2p - f) / s + 1) x floor((n + 2p - f) / s + 1)
- now for multi-channeled layers: n x n x n_c * f x f x n_c = n-f+1 x n-f+1 x n_c' where n_c is the no. of channels
- pooling n x n x n_c => floor((n - f) /s + 1) x floor((n - f) /s + 1)
'''