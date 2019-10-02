import numpy as np

def model_config_test():
    np.random.seed(2019) # for reproducibility
    X = np.random.randn(8,4)
    Y = np.random.randn(8,2)
    return X, Y, [3]

def linear_fwd_test():
    np.random.seed(2019) # for reproducibility
    A = np.random.randn(2,3)
    W = np.random.randn(3,1)
    b = np.random.randn(1,1)
    return W, b, A

def singlelayer_fwd_test():
    np.random.seed(2019) # for reproducibility
    A_prev = np.random.randn(2,3)
    W = np.random.randn(3,1)
    b = np.random.randn(1,1)
    return W, b, A_prev

def forward_prop_test():
    np.random.seed(2019) # for reproducibility
    X = np.random.randn(4,5)
    params = {}
    params['W1'] = np.random.randn(5,4)
    params['b1'] = np.random.randn(1,4)
    params['W2'] = np.random.randn(4,3)
    params['b2'] = np.random.randn(1,3)
    params['W3'] = np.random.randn(3,1)
    params['b3'] = np.random.randn(1,1)
    Y = (np.random.randn(4,1) > 0) * 1.
    return X, Y, params

def linear_back_test():
    np.random.seed(2019) # for reproducibility
    dZ = np.random.randn(2,1)
    cache = {}
    cache['A_prev'] = np.random.randn(2,3)
    cache['W'] = np.random.randn(3,1)
    cache['b'] = np.random.randn(1,1)
    return dZ, cache

def non_linearity_test():
    np.random.seed(2019) # for reproducibility
    dA = np.random.randn(2,3)
    cache = {}
    cache['Z'] = np.random.randn(2,3)
    return dA, cache

def singlelayer_back_test():
    np.random.seed(2019)
    dA = np.random.randn(2,1)
    A_prev = np.random.randn(2,3)
    W = np.random.randn(3,1)
    b = np.random.randn(1,1)
    Z = np.random.randn(2,1)
    cache = {}
    cache['LINEAR'] = {'W': W, 'b': b, 'A_prev': A_prev}
    cache['ACTIVATION'] = {'Z': Z}    
    return dA, cache

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