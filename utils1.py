import numpy as np

def sigmoid(x):
    s = 1 / ( 1 + np.exp(-x))
    return s

def model_config_test():
    np.random.seed(2019) # for reproducibility
    return np.random.randn(100, 2), np.random.randn(100, 1), 8

def forward_prop_test():
    np.random.seed(2019) # for reproducibility
    X = np.random.randn(3, 2)
    Y = (np.random.randn(3, 1) >= 0) * 1.
    params = {'W1': np.random.randn(2, 5), 'b1': np.random.randn(1, 5),
              'W2': np.random.randn(5, 1), 'b2': np.random.randn(1,1)}
    return params, X, Y

def back_prop_test():
    np.random.seed(2019) # for reproducibility
    X = np.random.randn(3, 2)
    Y = (np.random.randn(3, 1) >= 0) * 1.
    params = {'W1': np.random.randn(2, 5), 'b1': np.random.randn(1, 5), 
              'W2': np.random.randn(5, 1), 'b2': np.random.randn(1,1)}
    cache = {'A1': np.random.randn(3, 5), 'A2': np.random.randn(3, 1),
             'Z1': np.random.randn(3, 5),'Z2': np.random.randn(2,1)}
    return params, X, Y, cache

def update_params_test():
    np.random.seed(2019) # for reproducibility
    params = {'W1': np.random.randn(2, 5), 'b1': np.random.randn(1, 5),
              'W2': np.random.randn(5, 1), 'b2': np.random.randn(1,1)}
    grads = {'dW1': np.random.randn(2, 5), 'db1': np.random.randn(1, 5),
             'dW2': np.random.randn(5, 1), 'db2': np.random.randn(1,1)}
    return params, grads

def model_predict_test():
    np.random.seed(2019) # for reproducibility
    X = np.random.randn(3, 2)
    params = {'W1': np.array([[ 5.82809, -4.5018,   0.15839,  0.02176, -0.62088],
                              [ 3.95058, -1.81251,  2.21641,  0.47214,  0.75909]]),
              'b1': np.array([[-1.26981,  0.0359,  -0.83395, -3.19964,  1.01775]]),
              'W2': np.array([[ 7.87515],[-4.7095 ],[ 1.23063],[ 0.18959],[-0.29637]]),
              'b2': np.array([[-0.52574]])}
    return params, X