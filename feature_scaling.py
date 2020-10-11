import numpy as np

def feature_scaling(X):
    '''
    Applies feature scaling on a given matrix
    :param X:
    :return: triplet [X_scaled, lmbda, mu] where X_scaled is
    the modified matrix, lmbda is the mean average and mu is the standard deviation
    We will need this last two values for when we scale the test data set's features
    '''
    img_size = X.shape[1]
    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]));
    lmbda = np.mean(X, axis=1, keepdims=True)  # Computing the mean average of each input
    mu = np.std(X)  # Computing the standard deviation
    X_scaled = (X - lmbda) / mu  # Scaling features
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], img_size, img_size, 1))
    return [X_scaled, lmbda, mu]