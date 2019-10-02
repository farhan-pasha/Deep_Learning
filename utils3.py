import numpy as np
import h5py

import matplotlib.pyplot as plt

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
    # sigmoid output layer
    dZK = AK - Y
    grads["dW{}".format(K)], grads["db{}".format(K)], grads["dA{}".format(K)] = linear_back(dZK, cache['LINEAR'])
    for k in reversed(range(K - 1)):
        cache = caches[k]
        grads["dW{}".format(k + 1)], grads["db{}".format(k + 1)], grads["dA{}".format(k + 1)] = singlelayer_back(grads["dA{}".format(k + 2)], cache, 'ReLU')
    return grads

def forward_prop_test():
    np.random.seed(2019) # for reproducibility
    X = np.random.randn(1,5)
    params = {}
    params['W1'] = np.random.randn(5,4)
    params['b1'] = np.random.randn(1,4)
    params['W2'] = np.random.randn(4,3)
    params['b2'] = np.random.randn(1,3)
    params['W3'] = np.random.randn(3,1)
    params['b3'] = np.random.randn(1,1)
    Y = (np.random.randn(1,1) > 0) * 1.
    return X, Y, params

def params2theta(params):
    theta = np.array([])
    for key in params.keys():
        theta = np.concatenate((theta, params[key].reshape(-1)))
    return theta

def theta2params(theta, params):
    new_params = {}
    offset = 0
    for key in params.keys():
        shape = params[key].shape 
        count = np.prod(shape)
        new_params[key] = theta[offset:offset + count].reshape(shape)
        offset = count + offset
    return new_params

def update_params(params, grads, learning_rate=0.8):
    K = len(params) >> 1
    for k in range(1, K + 1):
        params['W{}'.format(k)] = params['W{}'.format(k)] - learning_rate * grads['dW{}'.format(k)] 
        params['b{}'.format(k)] = params['b{}'.format(k)] - learning_rate * grads['db{}'.format(k)]     
    return params

def model_predict(params, X):
    AK, _, _ = forward_prop(params, X)
    Y_hat = (AK > 0.5) * 1.
    return Y_hat

def model_config(X, Y, n_h=[128, 64, 16, 8]):
    n_x = X.shape[1] 
    n_y = Y.shape[1]
    dims = sum([[n_x], n_h, [n_y]], [])
    return dims

def plot_model(model, X, Y):
    range0 = np.arange(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 0.05)
    range1 = np.arange(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 0.05)
    XX, YY = np.meshgrid(range0, range1)
    Y_hat = model(np.c_[XX.ravel(), YY.ravel()])
    Y_hat = Y_hat.reshape(XX.shape)
    plt.contourf(XX, YY, Y_hat, cmap=plt.cm.Spectral)
    plt.ylabel('$x_2$')
    plt.xlabel('$x_1$')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)

def load_dataset():
    h5file = h5py.File('datasets/2Dlineardata.h5', "r")
    X_train = h5file["X_train"][:]
    Y_train = h5file["Y_train"][:]
    X_test = h5file["X_test"][:]
    Y_test = h5file["Y_test"][:]
    h5file.close()
    return (X_train, Y_train), (X_test, Y_test)

def back_prop_test():
    np.random.seed(2019) # for reproducibility
    AK = np.random.randn(2, 1)
    Y = np.array([[1, 0]]).T
    caches = []
    A1 = np.random.randn(2,4)
    W1 = np.random.randn(4,3)
    b1 = np.random.randn(1,3)
    Z1 = np.random.randn(2,3)
    caches.append({'LINEAR' : {'W': W1, 'b': b1, 'A_prev': A1}, 'ACTIVATION': {'Z': Z1}})
    A2 = np.random.randn(2,3)
    W2 = np.random.randn(3,1)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(2,1)
    caches.append({'LINEAR' : {'W': W2, 'b': b2, 'A_prev': A2}, 
                   'ACTIVATION': {'Z': Z2}})
    return AK, Y, caches

def forward_prop_with_dropout_test():
    np.random.seed(2019) # for reproducibility
    X = 0.1 * np.random.randn(1,500)
    params = {}
    params['W1'] = 0.2 * np.random.randn(500,400)
    params['b1'] = np.zeros((1,400))
    params['W2'] = 0.1 * np.random.randn(400,300)
    params['b2'] = np.zeros((1,300))
    params['W3'] = 0.1 * np.random.randn(300,200)
    params['b3'] = np.zeros((1,200))
    params['W4'] = 0.05 * np.random.randn(200,100)
    params['b4'] = np.zeros((1,100))
    Y = (np.random.rand(1,100) > 0.5) * 1.
    return X, Y, params, [0.5, 0.5, 0.3]

def backprop_with_dropout_test():
    np.random.seed(2019) # for reproducibility
    X = np.abs(8. * np.random.randn(1,8))
    params = {}
    params['W1'] = np.abs(8. * np.random.randn(8,3))
    params['b1'] = np.zeros((1,3))
    params['W2'] = np.abs(3. * np.random.randn(3,2))
    params['b2'] = np.zeros((1,2))
    params['W3'] = np.abs(np.random.randn(2,1))
    params['b3'] = np.random.randn(1,1)
    Y = (np.random.rand(1,1) > 0.5) * 1.
    return X, Y, params, [0.5, 0.5]