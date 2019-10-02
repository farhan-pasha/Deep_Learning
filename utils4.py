import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

def update_params_test():
    np.random.seed(2019) # for reproducibility
    params = {}
    params['W1'] = np.random.randn(4,3)
    params['b1'] = np.random.randn(1,3)
    params['W2'] = np.random.randn(3,1)
    params['b2'] = np.random.randn(1,1)
    np.random.seed(3)
    grads = {}
    grads['dW1'] = np.random.randn(4,3)
    grads['db1'] = np.random.randn(1,3)
    grads['dW2'] = np.random.randn(3,1)
    grads['db2'] = np.random.randn(1,1)
    return params, grads

def stochastic_mini_batch_test():
    np.random.seed(2019)
    mini_batch_sz = 64
    X = np.random.randn(132, 2)
    Y = np.random.randn(132, 1) > 0.5
    return X, Y, mini_batch_sz

def initialise_velocity_test():
    np.random.seed(2019)
    params = {}
    params['W1'] = np.random.randn(4,3)
    params['b1'] = np.random.randn(1,3)
    params['W2'] = np.random.randn(3,1)
    params['b2'] = np.random.randn(1,1)
    return params

def momentum_update_params_test():
    np.random.seed(2019) # for reproducibility
    params, grads = update_params_test()
    v = {}
    v['dW1'] = np.zeros((4,3))
    v['db1'] = np.zeros((1,3))
    v['dW2'] = np.zeros((3,1))
    v['db2'] = np.zeros((1,1))
    return params, grads, v

def adam_update_params_test():
    np.random.seed(2019) # for reproducibility
    params, grads, v = momentum_update_params_test()
    s = {}
    s['dW1'] = np.zeros((4,3))
    s['db1'] = np.zeros((1,3))
    s['dW2'] = np.zeros((3,1))
    s['db2'] = np.zeros((1,1))
    return params, grads, v, s

def linear_fwd(W, b, A):
    Z = A @ W + b
    cache = {"W": W, "b": b, "A_prev": A}
    return Z, cache

def sigmoid_fwd(Z):
    A = 1 / ( 1 + np.exp(-Z))
    cache = {"Z": Z}
    return A, cache

def relu_fwd(Z):
    A = np.maximum(0, Z)
    cache = {"Z": Z}
    return A, cache    

def singlelayer_fwd(W, b, A_prev, non_linearity='ReLU'):
    Z, linear_cache = linear_fwd(W, b, A_prev)
    if non_linearity == 'ReLU':
        A, activation_cache = relu_fwd(Z)
    elif non_linearity == 'Sigmoid':
        A, activation_cache = sigmoid_fwd(Z)
    return A, {'LINEAR': linear_cache, 'ACTIVATION': activation_cache}
    
def forward_prop(params, X, Y=None):
    caches = []
    K = len(params) >> 1
    A = X
    for k in range(1, K):
        A_prev = A
        W = params['W{}'.format(k)]
        b = params['b{}'.format(k)]
        A, cache = singlelayer_fwd(W, b, A_prev, non_linearity='ReLU')
        caches.append(cache)
    A_prev = A
    W = params['W{}'.format(K)]
    b = params['b{}'.format(K)]
    A, cache = singlelayer_fwd(W, b, A_prev, non_linearity='Sigmoid')
    caches.append(cache)
    loss = float('nan')
    if Y is not None:
        n = Y.shape[0]
        loss = np.multiply(-np.log(A),Y) + np.multiply(-np.log(1 - A), 1 - Y)
        loss = np.squeeze(np.nansum(loss)) / n

    return A, loss, caches

def model_predict(params, X):
    AK, _, _ = forward_prop(params, X)
    Y_hat = (AK > 0.5) * 1. 
    return Y_hat

def linear_back(dZ, cache):
    W = cache["W"]
    b = cache["b"]
    A_prev = cache["A_prev"]
    n = A_prev.shape[0]
    dW = (A_prev.T @ dZ) / n
    db = np.sum(dZ, axis=0, keepdims=True) / n
    dA_prev = dZ @ W.T # np.dot(dZ, W.T)
    return dW, db, dA_prev

def relu_back(dA, cache):
    Z = cache["Z"]
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_back(dA, cache):
    Z = cache["Z"]
    S = 1 / ( 1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ

def singlelayer_back(dA, cache, non_linearity='ReLU'):
    linear_cache = cache['LINEAR']
    activation_cache = cache['ACTIVATION']
    if non_linearity == 'Sigmoid':
        dZ = sigmoid_back(dA, activation_cache)
    elif non_linearity == 'ReLU':
        dZ = relu_back(dA, activation_cache)
    dW, db, dA_prev = linear_back(dZ, linear_cache)
    return dW, db, dA_prev

def back_prop(AK, Y, caches):
    grads = {}
    K = len(caches)
    n = AK.shape[0]
    cache = caches[K - 1]
    dZK = AK - Y
    grads["dW{}".format(K)], grads["db{}".format(K)], grads["dA{}".format(K)] = linear_back(dZK, cache['LINEAR'])
    for k in reversed(range(K - 1)):
        cache = caches[k]
        grads["dW{}".format(k + 1)], grads["db{}".format(k + 1)], grads["dA{}".format(k + 1)] = singlelayer_back(grads["dA{}".format(k + 2)], cache, 'ReLU')
    return grads

def model_config(X, Y, n_h=[128, 64, 16, 8]):
    n_x = X.shape[1] 
    n_y = Y.shape[1]
    dims = sum([[n_x], n_h, [n_y]], [])
    return dims

def he_init(dims):
    np.random.seed(3)
    K = len(dims)
    params = {}    
    for k in range(1, K):
        params['W{}'.format(k)] = np.random.randn(dims[k - 1], dims[k]) * math.sqrt(2. / dims[k-1])
        params['b{}'.format(k)] = np.zeros((1, dims[k]))
    return params

def plot_model(model, X, Y, title):
    range0 = np.arange(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 0.05)
    range1 = np.arange(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 0.05)
    XX, YY = np.meshgrid(range0, range1)
    Y_hat = model(np.c_[XX.ravel(), YY.ravel()])
    Y_hat = Y_hat.reshape(XX.shape)
    plt.contourf(XX, YY, Y_hat, cmap=plt.cm.Spectral)
    plt.ylabel('$x_2$')
    plt.xlabel('$x_1$')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.show()
