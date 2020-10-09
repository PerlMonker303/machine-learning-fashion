import numpy as np

def feature_scaling(X):
    '''
    Applies feature scaling on a given matrix
    :param X:
    :return: triplet [X_scaled, lmbda, mu] where X_scaled is
    the modified matrix, lmbda is the mean average and mu is the standard deviation
    We will need this last two values for when we scale the test data set's features
    '''
    lmbda = np.mean(X, axis=1, keepdims=True)  # Computing the mean average of each input
    mu = np.std(X)  # Computing the standard deviation
    X_scaled = (X - lmbda) / mu  # Scaling features
    return [X_scaled, lmbda, mu]