# Package imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils2 as u
#utils.py contains unit tests to check code


%matplotlib inline
np.set_printoptions(precision=5, suppress=True)

def load_dataset(filepath='datasets/catsvsnoncats.h5'):
    """
    Load and pre-process dataset
    
    Arguments:
    filepath -- string, dataset path

    Returns:
    (X_train, Y_train), (X_test, Y_test), classes -- training and test datasets
    """
    h5file = h5py.File(filepath, "r")
    X_train = np.array(h5file["X_train"][:])
    Y_train = np.array(h5file["Y_train"][:])
    X_test = np.array(h5file["X_test"][:])
    Y_test = np.array(h5file["Y_test"][:])
    classes = np.array(h5file["Classes"][:]) 
    h5file.close()
    # Reshape and scale datasets containing N_train and N_test mages such that
    # X_train.shape = (n_train, W * H * 3)
    # Y_train.shape = (n_train, 1)
    # X_test.shape = (n_test, W * H * 3)
    # X_test.shape = (n_test, 1)
    # Scale X_train and X_test from range {0..255} to (0,1)
    
    X_train = X_train.reshape(X_train.shape[0],-1)/255
    X_test = X_test.reshape(X_test.shape[0],-1)/255
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    Y_test = Y_test.reshape(Y_test.shape[0],1)

    return (X_train, Y_train), (X_test, Y_test), classes

def model_config(X, Y, n_h=[128, 64, 16, 8]):
    """
    Arguments:
    X -- n data samples, shape = (n, n_x)
    Y -- ground truth label, column vector of shape (n, n_y)
    n_h -- array with number of units in hidden layers, size K-1
    
    Returns:
    params -- dictionary containing initialised model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised bias vector of shape (1, n_h1)
        ...
        Wk -- initialised weight matrix of shape (n_hk-1, n_hk)
        bk -- initialised bias vector of shape (1, n_hk)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised bias vector of shape (1, n_y)
    """
    
    n_x = X.shape[1] # size of input layer
    n_y = Y.shape[1] # size of output layer
    
    dims = sum([[n_x], n_h, [n_y]], [])
    K = len(dims) # number of network layers

    params = {}
    
    
    for k in range(1, K):
        params['W{}'.format(k)] = (1/np.sqrt(dims[k-1])) * np.random.randn(dims[k-1],dims[k])
        params['b{}'.format(k)] = np.zeros((1,dims[k]))
                                                    
        assert(params['W{}'.format(k)].shape == (dims[k - 1], dims[k]))
        assert(params['b{}'.format(k)].shape == (1, dims[k]))

    assert(X.shape[0] == (Y.shape[0]))
    return params

X, Y, n_h = u.model_config_test()
params = model_config(X, Y, [3])
for k in range(1, len(params) // 2 + 1):
    print("W{} = {}".format(k, params['W{}'.format(k)]))
    print("b{} = {}".format(k, params['b{}'.format(k)]))

# Forward propagation for linearity
def linear_fwd(W, b, A):
    """
    Linearity

    Arguments:
    W -- weight matrix, shape (n_hk-1, n_hk)
    b -- bias row vector, shape (1, n_hk)
    A -- input, shape (n, n_hk-1)

    Returns:
    Z -- linear output, shape (n, n_hk)
    cache -- dictionary for backpropagation
        W -- weight matrix
        b -- bias row vector
        A_prev -- input
    """
    
    Z = A@W+b
    cache = {"W": W, "b": b, "A_prev": A}
    
    assert(Z.shape == (A.shape[0], W.shape[1]))
    return Z, cache

# Forward propagation for sigmoid non-linearity
def sigmoid_fwd(Z):
    """
    Sigmoid activation
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(Z), same shape as Z
    cache -- dictionary for backpropagation
        Z -- activation's input
    """

    A = 1.0/(1.0+np.exp(-Z))
    cache = {'Z': Z}
    
    assert(A.shape == Z.shape)
    return A, cache

# Forward propagation for ReLU non-linearity
def relu_fwd(Z):
    """
    RELU activation

    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of ReLU(Z), same shape as Z
    cache -- dictionary for backpropagation
        Z -- activation's input
    """
    
    A = np.maximum(Z, 0)
    cache =  {'Z': Z}
    
    assert(A.shape == Z.shape)
    return A, cache



W, b, A = u.linear_fwd_test()
Z, cache = linear_fwd(W, b, A)
print("Z.T = {}".format(Z.T))
A, cache = relu_fwd(Z)
print("A.T = ReLU(Z.T) = {}".format(A.T))
A, cache = sigmoid_fwd(Z)
print("A.T = sigmoid(Z.T) = {}".format(A.T))


# Single layer forward propagation for 
def singlelayer_fwd(W, b, A_prev, non_linearity='ReLU'):
    """
    Single layer forward propagation (linear + non-linearity)

    Arguments:
    W -- weight matrix, shape (n_hk-1, n_hk)
    b -- bias row vector, shape (1, n_hk)
    A_prev -- input, shape (n, n_hk-1)
    non_linearity -- string ('ReLU' or 'Sigmoid') activation for layer
    
    Returns:
    A -- output of g(A_prev @ W + b), shape (n, n_hk)
    cache -- dictionary for backprop
        LINEAR -- dictionary linear cache
        ACTIVATION -- dictionary activation cache        
    """
    
    Z, linear_cache = linear_fwd(W, b, A_prev)
    
    if non_linearity == 'ReLU':
        A, activation_cache = relu_fwd(Z)
    elif non_linearity == 'Sigmoid':
        A, activation_cache = sigmoid_fwd(Z)
    
    assert(A.shape == (A_prev.shape[0], W.shape[1]))
    return A, {'LINEAR': linear_cache, 'ACTIVATION': activation_cache}

W, b, A_prev = u.singlelayer_fwd_test()
A, cache = singlelayer_fwd(W, b, A_prev, 'Sigmoid')
print("A_Sigmoid.T = {}".format(A.T))
print("cache = {}".format(cache))
A, cache = singlelayer_fwd(W, b, A_prev, 'ReLU')
print("A_ReLU.T = {}".format(A.T))
print("cache = {}".format(cache))

# Forward propagation (inference)
def forward_prop(params, X, Y=None):
    """
    Compute the layer activations and loss if needed

    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised weight matrix of shape (1, n_h1)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised weight matrix of shape (1, n_y)
    X -- n data samples, shape = (n, n_x)
    Y -- optional argument, ground truth label, column vector of shape (n, n_y)

    Returns:
    A -- final layer output (activation value) 
    loss -- cross-entropy loss or NaN if Y=None
    caches -- array of caches for the K layers
    """
    
    caches = []
    K = len(params) >> 1
    A = X
    
    # K-1 [Linear->ReLU] layer
    for k in range(1, K):
        A_prev = A
        W = params["W{}".format(k)]
        b = params["b{}".format(k)]
        A, cache = singlelayer_fwd(W, b, A_prev)
        caches.append(cache)
    # 1 [Linear->Sigmoid] layer
    A_prev = A
    W = params["W{}".format(K)]
    b = params["b{}".format(K)]
    A, cache = singlelayer_fwd(W, b, A_prev,'Sigmoid')
    #print(A)
    caches.append(cache)
    
    loss = float('nan')
    if Y is not None:
        Y_hat = A
        n = Y.shape[0]
        # Compute the cross-entropy loss
        loss = -(1/n)*(Y.T@np.log(Y_hat)+(1-Y).T@np.log(1-Y_hat))

        loss = np.squeeze(loss)
        assert(loss.dtype == float)
        
    assert(A.shape == (X.shape[0], W.shape[1]))
    return A, loss, caches

X, Y, params = u.forward_prop_test()
A, loss, caches = forward_prop(params, X, Y)
print("A.T = {}".format(A.T))
print("loss = {:.5f}".format(loss))
print("{} caches".format(len(caches)))


# Backward propagation for linearity
def linear_back(dZ, cache):
    """
    Linearity backprop

    Arguments:
    dZ -- gradient of loss with respect to current layer linear output
    cache -- dictionary from forward propagation
        W -- weight matrix
        b -- bias row vector
        A_prev -- previous layer activation input

    Returns:
    dW -- gradient of loss with respect to current layer weights
    db -- gradient of loss with respect to current layer bias
    dA_prev -- gradient of loss with respect to activation of previous layer output
    """

    W = cache['W']
    b = cache['b']
    A_prev = cache['A_prev']
    n = A_prev.shape[0]
    dW = 1/n*(A_prev.T @ dZ)
    db = 1/n*(sum(dZ)).reshape(b.shape)
    dA_prev = dZ @ W.T
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dW, db, dA_prev

# Backward propagation for ReLU non-linearity
def relu_back(dA, cache):
    """
    ReLU backprop

    Arguments:
    dA -- gradient of loss with respect to activation
    cache -- dictionary from forward propagation
        Z -- layer linearity output 

    Returns:
    dZ -- gradient of loss with respect to Z
    """
    
    Z = cache['Z']
    dZ = dA*((Z>0)+0)
    
    assert(dZ.shape == Z.shape)
    return dZ

# Backward propagation for sigmoid non-linearity
def sigmoid_back(dA, cache):
    """
    Sigmoid backprop

    Arguments:
    dA -- gradient of loss with respect to activation
    cache -- dictionary from forward propagation
        Z -- layer linearity output 

    Returns:
    dZ -- gradient of loss with respect to Z
    """
    
    Z = cache['Z']
    S = 1/(1+np.exp(Z))
    dZ = dA*S*(1-S)
    
    assert(dZ.shape == Z.shape)
    return dZ

dZ, cache = u.linear_back_test()
dW, db, dA_prev = linear_back(dZ, cache)
print("dW.T = {}".format(dW.T))
print("db = {}".format(db))
print("dA_prev = {}".format(dA_prev))
A, cache = u.non_linearity_test()
dZ = relu_back(A, cache)
print("ReLU: dZ = {}".format(dZ))
dZ = sigmoid_back(A, cache)
print("Sigmoid: dZ = {}".format(dZ))

def singlelayer_back(dA, cache, non_linearity='ReLU'):
    """
    Single layer backprop (linear + non-linearity)

    Arguments:
    dA -- gradient of loss with respect to activation
    cache -- dictionary from forward propagation
        LINEAR -- dictionary from forward linear propagation 
        ACTIVATION -- dictionary from forward non-linearity propagation 
    non_linearity -- string ('ReLU' or 'Sigmoid') activation for layer

    Returns:
    dW -- gradient of loss with respect to current layer weights
    db -- gradient of loss with respect to current layer bias
    dA_prev -- gradient of loss with respect to activation of previous layer output
    """
    
    linear_cache = cache['LINEAR']
    activation_cache = cache['ACTIVATION']
    
    if non_linearity == 'Sigmoid':
        dZ = sigmoid_back(dA, activation_cache)
    elif non_linearity == 'ReLU':
        dZ = relu_back(dA, activation_cache)
        
    dW, db, dA_prev = linear_back(dZ, linear_cache)
    
    return dW, db, dA_prev

dA, cache = u.singlelayer_back_test()


dW, db, dA_prev = singlelayer_back(dA, cache, 'ReLU')
print("ReLU: dW.T = {}".format(dW.T))
print("ReLU: db = {}".format(db))
print("ReLU: dA_prev = {}".format(dA_prev))

dW, db, dA_prev = singlelayer_back(dA, cache, 'Sigmoid')
print("Sigmoid: dW.T = {}".format(dW.T))
print("Sigmoid: db = {}".format(db))
print("Sigmoid: dA_prev = {}".format(dA_prev))


# Backward_propagation
def back_prop(AK, Y, caches):
    """
    Compute back-propagation gradients
    
    Arguments:
    AK -- probability vector, final layer output, shape (1, n_y)
    Y -- ground truth output (n, n_y)
    caches -- array of layer cache, len=K
    
    Returns:
    grads -- dictionary containing your gradients with respect to all parameters
        dW1 -- weight gradient matrix of shape (n_x, n_h1)
        db1 -- bias gradient vector of shape (1, n_h1)
        ...
        dWK -- weight gradient matrix of shape (n_hK-1, n_y)
        dbK -- bias gradient vector of shape (1, n_y)
    """
    
    grads = {}
    K = len(caches)
    n = AK.shape[0]
    assert(Y.shape == AK.shape)

    dAK = -(Y/AK -((1-Y)/(1-AK)))
    cache = caches[K-1]
    grads["dW{}".format(K)], grads["db{}".format(K)],  grads["dA{}".format(K)] = singlelayer_back(dAK, cache, non_linearity='Sigmoid')
    for k in reversed(range(K - 1)):
        cache = caches[k]
        grads["dW{}".format(k + 1)], grads["db{}".format(k + 1)], grads["dA{}".format(k + 1)] = singlelayer_back(grads["dA{}".format(k+2)], cache)
        
    return grads

AK, Y, caches = u.back_prop_test()
grads = back_prop(AK, Y, caches)
print("dW1.T = {}".format(grads['dW1'].T))
print("db1 = {}".format(grads['db1']))
print("dA1 = {}".format(grads['dA1']))
print("dW2.T = {}".format(grads['dW2'].T))
print("db2 = {}".format(grads['db2']))
print("dA2 = {}".format(grads['dA2']))


# Update model parameters
def update_params(params, grads, learning_rate=0.8):
    """
    Updates parameters using the gradient descent
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised weight matrix of shape (1, n_h1)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised weight matrix of shape (1, n_y)
    grads -- dictionary containing gradients
        dW1 -- weight gradient matrix of shape (n_x, n_h1)
        db1 -- bias gradient vector of shape (1, n_h1)
        ...
        dWK -- weight gradient matrix of shape (n_hK-1, n_y)
        dbK -- bias gradient vector of shape (1, n_y)
    learning_rate -- learning rate of the gradient descent (hyperparameter)

    Returns:
    params -- dictionary containing updated parameters
    """

    K = len(params) >> 1
    for k in range(1, K + 1):
        params['W{}'.format(k)] = params['W{}'.format(k)] - learning_rate * grads['dW{}'.format(k)]
        params['b{}'.format(k)] = params['b{}'.format(k)] - learning_rate * grads['db{}'.format(k)]
    
    return params

params, grads = u.update_params_test()
params = update_params(params, grads, 0.1)
print("W1.T = {}".format(params['W1'].T))
print("b1 = {}".format(params['b1']))
print("W2.T = {}".format(params['W2'].T))
print("b2 = {}".format(params['b2']))


# Parameter optimisation using backprop
def model_fit(params, X, Y, epochs=2500, learning_rate=0.0075, verbose=False):
    """
    Optimise model parameters by performing gradient descent
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised weight matrix of shape (1, n_h1)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised weight matrix of shape (1, n_y)
    X -- n data samples  (n, n_x)
    Y -- groud truth label vector of size (n, n_y)
    epochs -- number of iteration updates through dataset
    learning_rate -- learning rate of the gradient descent
    
    Returns:
    params -- dictionary with optimised parameters
    grads -- dictionary with final gradients
    loss_log -- list of loss values for every 100 updates
    """
    
    loss_log = []
    for i in range(epochs):
        A, loss, caches = forward_prop(params,X,Y)
        grads = back_prop(A, Y, caches)
        params = update_params(params, grads,learning_rate)
        
        # logs
        if i % 100 == 0:
            loss_log.append(np.asscalar(loss))
            if verbose:
                print("Loss after {} epochs: {:.3f}".format(i, loss))
     
    return params, grads, loss_log

X, Y, params = u.forward_prop_test()
params, grads, loss_log = model_fit(params, X, Y, epochs = 300, verbose=False)
print("loss = {}".format(np.array(loss_log)))
print("W1.T = {}".format(params['W1'].T))
print("b1 = {}".format(params['b1']))
print("W2.T = {}".format(params['W2'].T))
print("b2 = {}".format(params['b2']))
print("W3.T = {}".format(params['W3'].T))
print("b3 = {}".format(params['b3']))
print("dW1.T = {}".format(grads['dW1'].T))
print("db1 = {}".format(grads['db1']))
print("dW2.T = {}".format(grads['dW2'].T))
print("db2 = {}".format(grads['db2']))
print("dW3.T = {}".format(grads['dW3'].T))
print("db3 = {}".format(grads['db3']))



# Model inference
def model_predict(params, X):
    """
    Predict class label using model parameters
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised weight matrix of shape (1, n_h1)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised weight matrix of shape (1, n_y)
    X -- n data samples, shape (n, n_x)
    
    Returns:
    Y_hat -- vector with class predictions for examples in X, shape (n, n_y)
    """
        
    AK, _, _ = forward_prop(params, X)
    Y_hat = (AK>0.5).astype(float) # Convert activations to {0,1} predictions
        
    n = X.shape[0]
    assert(Y_hat.shape == (n, 1))    
    return Y_hat

X, _, params = u.forward_prop_test()
params['b1'] = np.zeros((1,4))
Y_hat = model_predict(params, X)
print("predictions.T = {}".format(Y_hat.T))

# Deep model
def deep_model(X_train, Y_train, X_test, Y_test, hidden_layers=[21, 7, 3], epochs=2500, learning_rate=0.007, verbose=True):
    '''
    Build, train and evalaute the K-layer model
    (K-1) * [LINEAR -> RELU]  -> [LINEAR -> SIGMOID] 
    
    Arguments:
    X_train -- training set a numpy array of shape (n_train, n_x)
    Y_train -- training groud truth vector of size (n_train, n_y)
    X_test -- testing set a numpy array of shape (n_test, n_x)
    Y_test -- testing groud truth vector of size (n_test, n_y)
    hidden_layers -- array with number of units in hidden layers
    epochs -- number of iteration updates through dataset for training (hyperparameter)
    learning_rate -- learning rate of the gradient descent (hyperparameter)
    
    Returns:
    model -- dictionary 
        PARAMS -- parameters
        LOSS -- log of training loss
        GRADS -- final 
        ACC -- array with training and testing accuracies
        LR -- learning rate
    '''
    
    params = model_config(X_train, Y_train,hidden_layers)
    params, grads, loss = model_fit(params, X_train, Y_train, epochs, learning_rate, verbose)
    Y_hat_train = model_predict(params, X_train)
    Y_hat_test = model_predict(params, X_test)
    train_acc = 100*(1 - np.mean(np.abs(Y_hat_train-Y_train)))
    test_acc = 100*(1-np.mean(np.abs(Y_hat_test - Y_test)))

    print("{:.1f}% training acc.".format(train_acc))
    print("{:.1f}% test acc.".format(test_acc))
        
    return {"PARAMS": params, "LOSS": loss, "GRADS": grads, "ACC": [train_acc, test_acc], "LR": learning_rate}


(X_train, Y_train), (X_test, Y_test), classes = load_dataset('datasets/catsvsnoncats.h5')
np.random.seed(2019)
model = deep_model(X_train, Y_train, X_test, Y_test, hidden_layers=[7])

(X_train, Y_train), (X_test, Y_test), classes = load_dataset('datasets/catsvsnoncats.h5')
np.random.seed(2019)
model = deep_model(X_train, Y_train, X_test, Y_test, hidden_layers=[21, 9, 7], learning_rate=0.009)

plt.plot(model["LOSS"])
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.title("Learning rate = {}".format(model["LR"]))
plt.show()

params = model["PARAMS"]
parameter_count = 0
for key in params.keys():
    parameter_count = parameter_count + np.prod(params[key].shape)
print("Number of trainable parameters: {}".format(parameter_count))