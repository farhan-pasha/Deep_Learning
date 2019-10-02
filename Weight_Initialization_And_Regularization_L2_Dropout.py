# Package imports
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import utils3 as u

%matplotlib inline
np.set_printoptions(precision=3, suppress=True)

%load_ext autoreload
%autoreload 2

# Forward propagation for linear model
def linear_fwd(W, X):
    """
    Linearity forward

    Arguments:
    W -- weights, (n_x, 1)
    X -- input (n, n_x)

    Returns:
    Y -- linear output, shape (n, 1)
    cache -- dictionary for backpropagation
        W -- weights, (n_x, 1)
        X -- linear input, (n, n_x)
    """
    
    Y = X @ W
    cache = {'W': W, 'X': X}

    return Y, cache

# testing
np.random.seed(2019)
Y, _ = linear_fwd(np.random.randn(2,1), np.random.randn(1,2))
print("Y = {}".format(Y))


# Backward propagation for linearity
def linear_back(L, cache):
    """
    Linearity backprop

    Arguments:
    L -- loss
    cache -- dictionary from forward propagation
        W -- weights, (n_x, 1)
        X -- linear input, (n, n_x)

    Returns:
    dW -- gradient of L with respect to W, (n_x, 1)
    """

    X = cache['X']
    n = X.shape[0]
    dW = X.T/n          
        
    return dW

# testing
np.random.seed(2019)
W, X = (np.random.randn(2,1), np.random.randn(1,2))
L, cache = linear_fwd(W, X)
dW = linear_back(L, cache)
print("dW.T = {}".format(dW.T))


# Gradient verification
def check_linear_grads(W, X, epsilon=1e-5):
    """
    Gradient verification for simple linear function
    
    Arguments:
    W -- weights, (n_x, 1)
    X -- input (n, n_x)
    epsilon -- small scalar e.g. 1e-5
    
    Returns:
    diff -- difference between gradient appoximation and backprop evaluation
    """
    
    L, cache = linear_fwd(W, X)
    grad = linear_back(L, cache)

    grad_approx = np.zeros(grad.shape)
    for i in range(np.prod(W.shape)):  
        W_plus = W.copy()    
        W_minus = W.copy()        
        W_plus[i] = W[i]+epsilon
        W_minus[i] = W[i]-epsilon
        L_plus, _ = linear_fwd(W_plus, X)
        L_minus, _ = linear_fwd(W_minus, X)
        grad_approx[i] = (L_plus-L_minus)/(2*epsilon)
    diff = np.linalg.norm(grad - grad_approx) #########why L2 norm
    diff = diff / (np.linalg.norm(grad) + np.linalg.norm(grad_approx))  ##why L2 norm
    if diff > 1e-8:
        print("Gradient Implementation Error")
    
    return diff

# testing
np.random.seed(2019)
W, X = (np.random.randn(2,1), np.random.randn(1,2))
diff = check_linear_grads(W, X)
print("gradient diff = {:.3e}".format(diff))

def check_grads(params, grads, X, Y, epsilon=1e-6):
    """
    Gradient verification for deep neural network
    
    Arguments:
    params -- dictionary containing the model parameters
    grads -- dictionary with gradients
    X -- data sample
    Y -- ground truth label
    epsilon -- small scalar
    
    Returns:
    diff -- difference between gradient appoximation and backprop evaluation
    """
    
    theta = u.params2theta(params) # convert params to list of scalars
    grad_approx = np.zeros(theta.shape)
    for i in range(theta.shape[-1]):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] = theta[i] + epsilon
        theta_minus[i] = theta[i] - epsilon
        
        # convert list of scalars to params
        params_plus = u.theta2params(theta_plus, params) 
        params_minus = u.theta2params(theta_minus, params)
        
        _, L_plus, _ = u.forward_prop(params_plus, X, Y)
        _, L_minus, _ = u.forward_prop(params_minus, X, Y)
        grad_approx[i] = (L_plus-L_minus) / (2*epsilon)
        
    grad_flat = np.array([])
    for key in params.keys():
        grad_flat = np.concatenate((grad_flat, grads['d'+key].reshape(-1)))
    diff = np.linalg.norm(grad_flat - grad_approx)
    diff = diff / (np.linalg.norm(grad_flat) + np.linalg.norm(grad_approx))  
    
    if diff > 1e-7:
        print("Gradient Implementation Error")
    
    return diff


# testing
X, Y, params = u.forward_prop_test()
A, loss, caches = u.forward_prop(params, X, Y)
grads = u.back_prop(A, Y, caches)
diff = check_grads(params, grads, X, Y)
print("gradient diff = {:.3e}".format(diff))

# Parameter optimisation using backprop
def model_fit(X, Y, n_h=[20, 8], epochs=15000, learning_rate=0.01, verbose=False, init='he'):
    """
    Optimise model parameters by performing gradient descent
    
    Arguments:
    n_h -- array with number of units in hidden layers, size K-1
    X -- n data samples  (n, n_x)
    Y -- groud truth label vector of size (n, n_y)
    epochs -- number of iteration updates through dataset
    learning_rate -- learning rate of the gradient descent
    init -- string, either 'random', 'zeros' or 'he'
    
    Returns:
    params -- dictionary containing model parameters
    grads -- dictionary with final gradients
    loss_log -- list of loss values for every 100 updates
    """
    
    # returns array [n_x, n_h[0], ..., n_h[K-1], n_y]
    dims = u.model_config(X, Y, n_h)

    # create and initialise model parameters
    if init == 'random':
        params = random_init(dims)
    elif init == 'zeros':
        params = zeros_init(dims)
    elif init == 'he':
        params = he_init(dims)
    
    loss_log = []
    for i in range(epochs):
        A, loss, caches = u.forward_prop(params, X, Y) # Cost and gradient computation
        grads = u.back_prop(A, Y, caches)
        params = u.update_params(params, grads, learning_rate)        
        
        # logs
        if i % 100 == 0:
            loss_log.append(np.asscalar(loss))
            if verbose:
                print("Loss after {} epochs: {:.3f}".format(i, loss))
     
    return params, grads, loss_log


np.random.seed(2019)

# training set
X_train, Y_train = sklearn.datasets.make_moons(n_samples=512, noise=.8)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
# test set
X_test, Y_test = sklearn.datasets.make_moons(n_samples=256, noise=.8)
Y_test = Y_test.reshape(Y_test.shape[0], 1)

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.reshape(-1), s=20, cmap=plt.cm.Spectral);

def zeros_init(dims):
    """
    Arguments:
    dims -- array with number of units in each layer, size K
    
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
    
    K = len(dims) # number of layers
    params = {}    
    for k in range(1, K):
        params['W{}'.format(k)] = np.zeros([dims[k-1],dims[k]])
        params['b{}'.format(k)] = np.zeros([1,dims[k]])
                                                    
        assert(params['W{}'.format(k)].shape == (dims[k - 1], dims[k]))
        assert(params['b{}'.format(k)].shape == (1, dims[k]))

    return params


# testing
np.random.seed(2019)
params = zeros_init([3,2,1])
print("W1 = {}".format(params["W1"]))
print("b1 = {} ".format(params["b1"]))
print("W2 = {}".format(params["W2"]))
print("b2 = {}".format(params["b2"]))


params, grads, loss_log = model_fit(X_train, Y_train, init='zeros')
print("loss after {} epochs = {:.2f}".format(len(loss_log) * 100, loss_log[-1]))

Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1))


def random_init(dims):
    """
    Arguments:
    dims -- array with number of units in each layer, size K
    
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
    
    K = len(dims) # number of layers

    params = {}    
    for k in range(1, K):
        params['W{}'.format(k)] = 3*(np.random.randn(dims[k-1],dims[k]))
        params['b{}'.format(k)] = np.zeros([1,dims[k]])
                                                    
        assert(params['W{}'.format(k)].shape == (dims[k - 1], dims[k]))
        assert(params['b{}'.format(k)].shape == (1, dims[k]))

    return params


# testing
np.random.seed(2019)
params = random_init([3,2,1])
print("W1 = {}".format(params["W1"]))
print("b1 = {}".format(params["b1"]))
print("W2 = {}".format(params["W2"]))
print("b2 = {}".format(params["b2"]))


np.random.seed(2019)
params, grads, loss_log = model_fit(X_train, Y_train, init='random')
print("loss after {} epochs = {:.2f}".format(len(loss_log) * 100, loss_log[-1]))

plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.show()

Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

%matplotlib inline
u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1))


def he_init(dims):
    """
    Arguments:
    dims -- array with number of units in each layer, size K
    
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
    K = len(dims) # number of layers

    params = {}    
    for k in range(1, K):
        params['W{}'.format(k)] = (1/np.sqrt(dims[k-1])) * np.random.randn(dims[k-1],dims[k])
        params['b{}'.format(k)] = np.zeros([1,dims[k]])
                                                    
        assert(params['W{}'.format(k)].shape == (dims[k - 1], dims[k]))
        assert(params['b{}'.format(k)].shape == (1, dims[k]))

    return params


# testing
np.random.seed(2019)
params = he_init([3,2,1])
print("W1 = {}".format(params["W1"]))
print("b1 = {}".format(params["b1"]))
print("W2 = {}".format(params["W2"]))
print("b2 = {}".format(params["b2"]))


np.random.seed(2019)
params, grads, loss_log = model_fit(X_train, Y_train, init='he')
print("loss after {} epochs = {:.2f}".format(len(loss_log) * 100, loss_log[-1]))

plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.show()

Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1))


np.random.seed(2019)
params, grads, loss_log = model_fit(X_train, Y_train, n_h=[18, 9, 9], epochs=30000, learning_rate=0.1, verbose=False, init='he')
print("loss after {} epochs = {:.2f}".format(len(loss_log) * 100, loss_log[-1]))

plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.show()

Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1))


# Parameter optimisation using regularisation
def regularised_model_fit(X, Y, n_h, lambd, epochs, learning_rate, verbose=False, init="he"):
    """
    Optimise model parameters by performing gradient descent
    
    Arguments:
    n_h -- array with number of units in hidden layers, size K-1
    X -- n data samples  (n, n_x)
    Y -- groud truth label vector of size (n, n_y)
    lambd -- regularisation scalar
    epochs -- number of iteration updates through dataset
    learning_rate -- learning rate of the gradient descent
    init -- string, either 'random', 'zeros' or 'he'
    
    Returns:
    params -- dictionary containing model parameters
    grads -- dictionary with final gradients
    loss_log -- list of loss values for every 100 updates
    """
    
    # returns array [n_x, n_h[0], ..., n_h[K-1], n_y]
    dims = u.model_config(X, Y, n_h)

    # create and initialise model parameters
    if init == 'random':
        params = random_init(dims)
    elif init == 'zeros':
        params = zeros_init(dims)
    elif init == 'he':
        params = he_init(dims)
    
    loss_log = []
    K = len(params) >> 1
    n = Y.shape[0]
    for i in range(epochs):
        A, loss, caches = u.forward_prop(params, X, Y)
        
        for k in range(1, K):  
            Wk = params["W{}".format(k)]
            loss = loss + (1/n)*(lambd/2) * np.sum(np.square(Wk))
        grads = regularised_back_prop(A, Y, caches, lambd)
        params = u.update_params(params, grads, learning_rate)        
        
        # logs
        if i % 100 == 0:
            loss_log.append(np.asscalar(loss))
            if verbose:
                print("Loss after {} epochs: {:.3f}".format(i, loss))
     
    return params, grads, loss_log


def regularised_back_prop(AK, Y, caches, lambd):
    """
    Compute back-propagation gradients with regularisation
    
    Arguments:
    AK -- probability vector, final layer output, shape (1, n_y)
    Y -- ground truth output (n, n_y)
    caches -- array of layer cache, len=K
    lambd -- regularisation scalar

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
    cache = caches[K - 1]
    
    ################### stable backprop for sigmoid output layer
    dZK = AK - Y
    grads["dW{}".format(K)], grads["db{}".format(K)], grads["dA{}".format(K)] = regularised_linear_back(dZK, cache['LINEAR'], lambd)

    for k in reversed(range(K - 1)):
        cache = caches[k]
        grads["dW{}".format(k + 1)], grads["db{}".format(k + 1)], grads["dA{}".format(k + 1)] = regularised_singlelayer_back(grads["dA{}".format(k + 2)], cache, 'ReLU', lambd)
    return grads


def regularised_singlelayer_back(dA, cache, non_linearity='ReLU', lambd=0):
    """
    Single layer backprop (linear + non-linearity) with regularisation

    Arguments:
    dA -- gradient of loss with respect to activation
    cache -- dictionary from forward propagation
        LINEAR -- dictionary from forward linear propagation 
        ACTIVATION -- dictionary from forward non-linearity propagation 
    non_linearity -- string ('ReLU' or 'Sigmoid') activation for layer
    lambd -- regularisation scalar

    Returns:
    dW -- gradient of loss with respect to current layer weights
    db -- gradient of loss with respect to current layer bias
    dA_prev -- gradient of loss with respect to activation of previous layer output
    """
        
    linear_cache = cache['LINEAR']
    activation_cache = cache['ACTIVATION']
    if non_linearity == 'Sigmoid':
        dZ = u.sigmoid_back(dA, activation_cache)
    elif non_linearity == 'ReLU':
        dZ = u.relu_back(dA, activation_cache)
    dW, db, dA_prev = regularised_linear_back(dZ, linear_cache, lambd)
    return dW, db, dA_prev


def regularised_linear_back(dZ, cache, lambd):
    """
    Linearity backprop

    Arguments:
    dZ -- gradient of loss with respect to current layer linear output
    cache -- dictionary from forward propagation
        W -- weight matrix
        b -- bias row vector
        A_prev -- previous layer activation input
    lambd -- regularisation scalar

    Returns:
    dW -- gradient of loss with respect to current layer weights
    db -- gradient of loss with respect to current layer bias
    dA_prev -- gradient of loss with respect to activation of previous layer output
    """
    dW, db, dA_prev = u.linear_back(dZ, cache)
    A_prev = cache["A_prev"]
    W = cache["W"]
    n = A_prev.shape[0]
    dW = dW + (lambd/n)*W
    return dW, db, dA_prev


# testing
AK, Y, caches = u.back_prop_test()
grads = regularised_back_prop(AK, Y, caches, 0.7)
print("dW1.T = {}".format(grads['dW1'].T))
print("dW2.T = {}".format(grads['dW2'].T))


np.random.seed(2019)
params, grads, loss_log = regularised_model_fit(X_train, Y_train, lambd=1.0, n_h=[18, 9, 9], epochs=30000, learning_rate=0.1, verbose=False, init='he')

plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.show()

print("loss after {} epochs = {:.2f}".format(len(loss_log) *100, loss_log[-1]))
Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1))


def forward_prop_with_dropout(params, X, Y=None, dropouts=None):
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
    dropouts -- optional argument, list of dropout rates from A1 to AK-1

    Returns:
    A -- final layer output (activation value) 
    loss -- cross-entropy loss or NaN if Y=None
    caches -- array of caches for the K layers
    """
        
    caches = []
    K = len(params) >> 1
    A = X
    for k in range(1, K):
        A_prev = A
        W = params['W{}'.format(k)]
        b = params['b{}'.format(k)]
        A, cache = u.singlelayer_fwd(W, b, A_prev, non_linearity='ReLU')

        if dropouts is not None:
            dropout_rate = dropouts[k - 1]
            D = np.random.rand(A.shape[0],A.shape[1])    # initialize dropout matrix D = np.random.rand(..., ...)
            #print(D)
            #print(dropout_rate)
            D = np.where(D<dropout_rate,0.0,1.0)    # convert entries of D to 0 or 1
            A = A*D    # shut down neuron units of A
            A = A/(1-dropout_rate)    # scale value of active neurons units
            #print(A)
            cache['DROPOUT'] = D     # add dropout matrix to cache
        
        caches.append(cache)
        
    A_prev = A
    W = params['W{}'.format(K)]
    b = params['b{}'.format(K)]
    A, cache = u.singlelayer_fwd(W, b, A_prev, non_linearity='Sigmoid')
    
    caches.append(cache)
    loss = float('nan')
    if Y is not None:
        n = Y.shape[0]
        loss = np.multiply(-np.log(A),Y) + np.multiply(-np.log(1 - A), 1 - Y)
        loss = np.squeeze(np.nansum(loss) / n)
    return A, loss, caches



# testing
X, Y, params, dropouts = u.forward_prop_with_dropout_test()
A, loss, caches = forward_prop_with_dropout(params, X, Y, dropouts)
for i, cache in enumerate(caches[0:len(dropouts)]):
        print("Layer {} dropout = {:.2f} (target {:.2f})".format(i + 1, 1 - np.mean(cache['DROPOUT']), dropouts[i]))


def back_prop_with_dropout(AK, Y, caches, dropouts=None):
    """
    Compute back-propagation gradients with dropout
    
    Arguments:
    AK -- probability vector, final layer output, shape (1, n_y)
    Y -- ground truth output (n, n_y)
    caches -- array of layer cache, len=K
    dropouts -- optional argument, list of dropout rates (K-1)
    
    Returns:
    grads -- dictionary containing your gradients with respect to all parameters
        dW1 -- weight gradient matrix of shape (n_x, n_h1)
        db1 -- bias gradient vector of shape (1, n_h1)
        ...
        dWK -- weight gradient matrix of shape (n_hK-1, n_y)
        dbK -- bias gradient vector of shape (1, n_y)
    """
    if dropouts is not None:
        dropout_cache = {}
        # retrieve dropout cache
        for i, cache in enumerate(caches[0:len(dropouts)]): # no dropout on output layer
            dropout_cache["D{}".format(i + 1)] = cache['DROPOUT']
        
    grads = {}
    K = len(caches)
    n = AK.shape[0]
    cache = caches[K - 1]
    # stable backprop for sigmoid output layer
    dZK = AK - Y
    grads["dW{}".format(K)], grads["db{}".format(K)], grads["dA{}".format(K)] = u.linear_back(dZK, cache['LINEAR'])
    
    for k in reversed(range(K - 1)):
        cache = caches[k]
        dA = grads["dA{}".format(k + 2)]
                
        # Hidden layer dropout
        if dropouts is not None:
            dropout_rate = dropouts[k]
            dA = dA*dropout_cache["D{}".format(k+1)]  # shut down the same units as during forward propagation
            dA = dA/(1-dropout_rate)    # scale va1ue of active neuron units
        
        grads["dW{}".format(k + 1)], grads["db{}".format(k + 1)], grads["dA{}".format(k + 1)] = u.singlelayer_back(dA, cache, 'ReLU')
    return grads


# testing
X, Y, params, dropouts = u.backprop_with_dropout_test()
A, loss, caches = forward_prop_with_dropout(params, X, Y, dropouts)
grads = back_prop_with_dropout(A, Y, caches, dropouts)
print("dW1.T = {}".format(grads['dW1'].T))
print("db1 = {}".format(grads['db1']))
print("dA1 = {}".format(grads['dA1']))
print("dW2.T = {}".format(grads['dW2'].T))
print("db2 = {}".format(grads['db2']))
print("dA2 = {}".format(grads['dA2']))


# Parameter optimisation using regularisation
def dropout_model_fit(X, Y, n_h, dropouts, epochs, learning_rate, verbose=False, init="he"):
    """
    Optimise model parameters by performing gradient descent
    
    Arguments:
    X -- n data samples  (n, n_x)
    Y -- groud truth label vector of size (n, n_y)
    n_h -- array with number of units in hidden layers, size K-1
    dropouts -- optional argument, list of dropout rates (K-1)
    epochs -- number of iteration updates through dataset
    learning_rate -- learning rate of the gradient descent
    init -- string, either 'random', 'zeros' or 'he'
    
    Returns:
    params -- dictionary containing model parameters
    grads -- dictionary with final gradients
    loss_log -- list of loss values for every 100 updates
    """
    
    # model topology, array [n_x, n_h[0], ..., n_h[K-1], n_y]
    dims = u.model_config(X, Y, n_h)

    # create and initialise model parameters
    if init == 'random':
        params = random_init(dims)
    elif init == 'zeros':
        params = zeros_init(dims)
    elif init == 'he':
        params = he_init(dims)
    
    loss_log = []
    K = len(params) >> 1
    n = Y.shape[0]
    for i in range(epochs):
        A, loss, caches = forward_prop_with_dropout(params, X, Y, dropouts)
        grads = back_prop_with_dropout(A, Y, caches, dropouts)
        params = u.update_params(params, grads, learning_rate)        
        
        # logs
        if i % 100 == 0:
            loss_log.append(np.asscalar(loss))
            if verbose:
                print("Loss after {} epochs: {:.3f}".format(i, loss))
     
    return params, grads, loss_log



np.random.seed(2019)
params, grads, loss_log = dropout_model_fit(X_train, Y_train, dropouts=[0.5, 0.3, 0.3], n_h=[18, 9, 9], epochs=30000, learning_rate=0.1, verbose=False, init='he')

plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.show()

print("loss after {} epochs = {:.2f}".format(len(loss_log) *100, loss_log[-1]))
Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1))





