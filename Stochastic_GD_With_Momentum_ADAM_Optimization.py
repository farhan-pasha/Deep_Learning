# Package imports
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import utils4 as u

%matplotlib inline
np.set_printoptions(precision=3, suppress=True)

%load_ext autoreload
%autoreload 2

# Update model parameters
def gd_update_params(params, grads, learning_rate=0.8):
    """
    Updates parameters using gradient descent
    
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
        ### INPUT YOUR CODE HERE ### (2 lines)
        params['W{}'.format(k)]-=learning_rate*grads['dW{}'.format(k)] 
        params['b{}'.format(k)]-=learning_rate*grads['db{}'.format(k)] 
        ### END OF YOUR CODE SEGMENT ### 
    
    return params

# testing
params, grads = u.update_params_test()
params = gd_update_params(params, grads, 0.1)
print("W1.T = {}".format(params['W1'].T))
print("b1 = {}".format(params['b1']))
print("W2.T = {}".format(params['W2'].T))
print("b2 = {}".format(params['b2']))

# BGD optimisation
for i in range(epochs):
   A, loss, caches = forward_prop(params, X[j,:], Y[j,:])
   grads = back_prop(A, Y[j,:], caches)
   params = gd_update_params(params, grads, learning_rate)        

# SGD optimisation    
for i in range(epochs):
    for j in range(X.shape[0]):
        A, loss, caches = forward_prop(params, X[j,:], Y[j,:])
        grads = back_prop(A, Y[j,:], caches)
        params = gd_update_params(params, grads, learning_rate)        

# Mini-batch GD optimisation    
for i in range(epochs):
    mini_bathes = stochastic_mini_batches(X, Y, mini_batch_sz, seed)
    for j in range(len(mini_batches)):
        (X_mini_bath, Y_mini_bath) = mini_batches[j]
        A, loss, caches = forward_prop(params, X_mini_bath, Y_mini_bath)
        grads = back_prop(A, Y_mini_bath, caches)
        params = gd_update_params(params, grads, learning_rate)

# Shuffle and partition a training set into mini-batches
def stochastic_mini_batches(X, Y, mini_batch_sz=128, seed=2019):
    """
    Creates a list of random mini-batches
    
    Arguments:
    X -- training set a numpy array of shape (n, n_x)
    Y -- training groud truth vector of size (n, n_y)
    mini_batch_sz -- size of the mini-batches, integer
    seed -- random seed for reproducibility
    
    Returns:
    mini_batches -- list of mini-batches [(X_mini_batch, Y_mini_batch), ..., (X_mini_batch, Y_mini_batch)]
    """
    
    np.random.seed(seed)            
    n = X.shape[0]
    mini_batches = []
        
    # Step 1: random permutation of (X, Y)
    permutation = list(np.random.permutation(n))
    X_perm = X[permutation]
    Y_perm = Y[permutation]


    # Step 2: Partition
    count = int(math.floor(n / mini_batch_sz))
    for i in range(count):
        X_mini_batch = X_perm[i*mini_batch_sz:(i+1)*mini_batch_sz]
        Y_mini_batch = Y_perm[i*mini_batch_sz:(i+1)*mini_batch_sz]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append(mini_batch)
    
    if n % mini_batch_sz != 0:
        X_mini_batch = X_perm[count*mini_batch_sz:]
        Y_mini_batch = Y_perm[count*mini_batch_sz:]
        mini_batch = (X_mini_batch,Y_mini_batch)
        mini_batches.append(mini_batch)

    return mini_batches

# testing
X, Y, mini_batch_sz = u.stochastic_mini_batch_test()
mini_batches = stochastic_mini_batches(X, Y, mini_batch_sz)

print("training set size = {}".format(X.shape[0]))
print("mini-batch size = {}".format(mini_batch_sz))
print("number of mini-batches = {}".format(len(mini_batches)))

import functools
print("number of examples in all mini-batches = {}".format(functools.reduce(lambda x, y : y[0].shape[0] + x , mini_batches, 0)))
print("number of examples in last mini-batch = {}".format(mini_batches[-1][0].shape[0]))


# Initialise momentum parameters
def initialise_velocity(params):
    """
    Initialises the optimiser velocity
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised weight matrix of shape (1, n_h1)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised weight matrix of shape (1, n_y)
    
    Returns:
    v -- dictionary containing current velocity
        dW1 -- zero matrix of shape W1
        db1 -- zero matrix of shape b1
        ...
        dWK -- zero matrix of shape WK
        dbK -- zero matrix of shape WK
    """
    
    K = len(params) >> 1
    v = {}
    
    # Initialize velocity
    for k in range(1, K + 1):
        v['dW{}'.format(k)] = np.zeros(list(params['W{}'.format(k)].shape))
        v['db{}'.format(k)] = np.zeros(list(params['b{}'.format(k)].shape))
        
    return v

# testing
params = u.initialise_velocity_test()
v = initialise_velocity(params)
print("v[\'dW1\'] = {}".format(v['dW1']))
print("v[\'db1\'] = {}".format(v['db1']))
print("v[\'dW2\'].T = {}".format(v['dW2'].T))
print("v[\'db2\'] = {}".format(v['db2']))

# Update model parameters using momentum
def momentum_update_params(params, grads, v, beta=0.9, learning_rate=0.8):
    """
    Updates parameters using gradient descent with momentum
    
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
    v -- dictionary containing current velocity
        dW1 -- momentum matrix of shape (n_x, n_h1)
        db1 -- momentum vector of shape (1, n_h1)
        ...
        dWK -- momentum matrix of shape (n_hK-1, n_y)
        dbK -- momentum vector of shape (1, n_y)
    beta -- momentum scalar (hyperparameter)
    learning_rate -- learning rate of the gradient descent (hyperparameter)

    Returns:
    params -- dictionary containing updated parameters
    v -- dictionary containing updated velocities
    """

    K = len(params) >> 1
    
    for k in range(1, K + 1):
        v['dW{}'.format(k)] = beta*v['dW{}'.format(k)] + (1-beta)*grads['dW{}'.format(k)]
        v['db{}'.format(k)] = beta*v['db{}'.format(k)] + (1-beta)*grads['db{}'.format(k)]
        params['W{}'.format(k)] = params['W{}'.format(k)] - learning_rate*v['dW{}'.format(k)]
        params['b{}'.format(k)] = params['b{}'.format(k)] - learning_rate*v['db{}'.format(k)]
    
    return params, v

params, grads, v = u.momentum_update_params_test()
params, v = momentum_update_params(params, grads, v, beta=0.9, learning_rate=0.01)
print("W1 = {}".format(params['W1']))
print("b1 = {}".format(params['b1']))
print("W2.T = {}".format(params['W2'].T))
print("b2 = {}".format(params['b2']))
print("v[\'dW1\'] = {}".format(v['dW1']))
print("v[\'db1\'] = {}".format(v['db1']))
print("v[\'dW2\'].T = {}".format(v['dW2'].T))
print("v[\'db2\'] = {}".format(v['db2']))

# Initialise adam parameters
def initialise_moments(params):
    """
    Initialises the optimiser 1st and 2nd moments
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h1)
        b1 -- initialised weight matrix of shape (1, n_h1)
        ...
        WK -- initialised weight matrix of shape (n_hK-1, n_y)
        bK -- initialised weight matrix of shape (1, n_y)
    
    Returns:
    v -- dictionary containing current 1st moment estimates
        dW1 -- zero matrix of shape W1
        db1 -- zero matrix of shape b1
        ...
        dWK -- zero matrix of shape WK
        dbK -- zero matrix of shape WK
    s -- dictionary containing current 2nd moment estimates
        dW1 -- zero matrix of shape W1
        db1 -- zero matrix of shape b1
        ...
        dWK -- zero matrix of shape WK
        dbK -- zero matrix of shape WK
    """
    
    K = len(params) >> 1
    v = {}
    s = {}
    
    # Initialize velocities
    for k in range(1, K + 1):
=        v['dW{}'.format(k)] = np.zeros(list(params['W{}'.format(k)].shape))
        v['db{}'.format(k)] = np.zeros(list(params['b{}'.format(k)].shape))
        s['dW{}'.format(k)] = np.zeros(list(params['W{}'.format(k)].shape))
        s['db{}'.format(k)] = np.zeros(list(params['b{}'.format(k)].shape))
=        
    return v, s



# testing
params = u.initialise_velocity_test()
v, s = initialise_moments(params)
print("v[\'dW1\'] = {}".format(v['dW1']))
print("v[\'db1\'] = {}".format(v['db1']))
print("v[\'dW2\'].T = {}".format(v['dW2'].T))
print("v[\'db2\'] = {}".format(v['db2']))
print("s[\'dW1\'] = {}".format(s['dW1']))
print("s[\'db1\'] = {}".format(s['db1']))
print("s[\'dW2\'].T = {}".format(s['dW2'].T))
print("s[\'db2\'] = {}".format(s['db2']))


# Update model parameters using ADAM
def adam_update_params(params, grads, v, s, t, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.8):
    """
    Updates parameters using Adam
    
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
    v -- dictionary containing current 1st moment estimates
        same keys as grads
    s -- dictionary containing current 2nd moment estimates
        same keys as grads
    t -- parameter update counter (integer)
    beta1 -- 1st moment estimate scalar (hyperparameter)
    beta2 -- 2nd moment estimate scalar (hyperparameter)
    epsilon -- small scalar for numerical stability (hyperparameter)
    learning_rate -- learning rate of the gradient descent (hyperparameter)

    Returns:
    params -- dictionary containing updated parameters
    v -- dictionary containing updated 1st moment estimates
    s -- dictionary containing updated 2nd moment estimates
    """

    K = len(params) >> 1
    v_bar = {}
    s_bar = {}
    
    for k in range(1, K + 1):
        v['dW{}'.format(k)] = beta1*v['dW{}'.format(k)] + (1-beta1)*grads['dW{}'.format(k)]
        v['db{}'.format(k)] = beta1*v['db{}'.format(k)] + (1-beta1)*grads['db{}'.format(k)]
        
        s['dW{}'.format(k)] = beta2*s['dW{}'.format(k)] + (1-beta2)*np.square(grads['dW{}'.format(k)])
        s['db{}'.format(k)] = beta2*s['dW{}'.format(k)] + (1-beta2)*np.square(grads['dW{}'.format(k)])

        v_bar['dW{}'.format(k)] = v['dW{}'.format(k)]/(1-np.power(beta1,t))
        v_bar['db{}'.format(k)] = v['db{}'.format(k)]/(1-np.power(beta1,t))

        s_bar['dW{}'.format(k)] = s['dW{}'.format(k)]/(1-np.power(beta2,t))
        s_bar['db{}'.format(k)] = s['db{}'.format(k)]/(1-np.power(beta2,t))
        
        params['W{}'.format(k)] = params['W{}'.format(k)] - (learning_rate*v_bar['dW{}'.format(k)]/(np.sqrt(s_bar['dW{}'.format(k)])+epsilon))
        params['b{}'.format(k)] = params['b{}'.format(k)] - (learning_rate*v_bar['db{}'.format(k)]/(np.sqrt(s_bar['db{}'.format(k)])+epsilon))
    
    return params, v, s


# testing
params, grads, v, s = u.adam_update_params_test()
params, v, s = adam_update_params(params, grads, v, s, t=3, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.01)
print("W1 = {}".format(params['W1']))
print("b1 = {}".format(params['b1']))
print("W2.T = {}".format(params['W2'].T))
print("b2 = {}".format(params['b2']))
print("v[\'dW1\'] = {}".format(v['dW1']))
print("v[\'db1\'] = {}".format(v['db1']))
print("v[\'dW2\'].T = {}".format(v['dW2'].T))
print("v[\'db2\'] = {}".format(v['db2']))
print("s[\'dW1\'] = {}".format(s['dW1']))
print("s[\'db1\'] = {}".format(s['db1']))
print("s[\'dW2\'].T = {}".format(s['dW2'].T))
print("s[\'db2\'] = {}".format(s['db2']))


# Parameter optimisation using different optimisers
def model_fit(X, Y, n_h=[20, 8], optimiser=None, epochs=15000, learning_rate=0.01, verbose=True):
    """
    Optimise model parameters by performing gradient descent
    
    Arguments:
    X -- n data samples  (n, n_x)
    Y -- groud truth label vector of size (n, n_y)
    n_h -- array with number of units in hidden layers, size K-1
    optimiser -- dictionary
        name: string, optimiser name 'gradient_descent', 'momentum' or 'adam'
        mini_batch_sz: int, size of mini-batch
        beta: scalar, required hyperparameter for momentum 
        beta1: scalar, required hyperparameter for adam 
        beta2: scalar, required hyperparameter for adam 
        epsilon: scalar, required hyperparameter for adam 
    epochs -- number of iteration updates through dataset
    learning_rate -- learning rate of the gradient descent
    
    Returns:
    params -- dictionary containing model parameters
    grads -- dictionary with final gradients
    loss_log -- list of loss values for every 100 updates
    """
    
    dims = u.model_config(X, Y, n_h)
    seed = 2019 # for reproducibility
    params = u.he_init(dims) # initialise model parameters
    loss_log = []
    
    # initialise optimiser
    mini_batch_sz = optimiser['mini_batch_sz']
    if optimiser['name'] == 'momentum':
        v = initialise_velocity(params)
        beta = optimiser['beta']
    elif optimiser['name'] == 'adam':
        v, s = initialise_moments(params)
        beta1 = optimiser['beta1']
        beta2 = optimiser['beta2']
        epsilon = optimiser['epsilon']
        t = 0

    for i in range(epochs):
        mini_batches = stochastic_mini_batches(X, Y, mini_batch_sz, seed)
        seed = seed + 1
        for j in range(len(mini_batches)):
            (X_mini_batch, Y_mini_batch) = mini_batches[j]
            
            A, loss, caches = u.forward_prop(params, X_mini_batch, Y_mini_batch) # Cost and gradient computation
            grads = u.back_prop(A, Y_mini_batch, caches) # Backprop
            
            # parameter update
            if optimiser['name'] == 'momentum':
                params,v = momentum_update_params(params, grads, v, beta, learning_rate) # Momentum
            elif optimiser['name'] == 'adam':
                t = t+1
                params,v,s = adam_update_params(params, grads, v, s, t, beta1, beta2, epsilon, learning_rate) # Adam
            else:
                params = gd_update_params(params, grads, learning_rate) # Gradient descent
            
        # logs
        if i % 100 == 0:
            loss_log.append(np.asscalar(loss))
        if verbose and (i == 0 or i % 1000 == 999):
            print("Loss after {} epoch{}: {:.5f}".format(i + 1, 's' if i > 0 else '', loss))
     
    return params, grads, loss_log

np.random.seed(2019)

# training set
X_train, Y_train = sklearn.datasets.make_moons(n_samples=512, noise=.2)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
# test set
X_test, Y_test = sklearn.datasets.make_moons(n_samples=256, noise=.2)
Y_test = Y_test.reshape(Y_test.shape[0], 1)

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.reshape(-1), s=20, cmap=plt.cm.Spectral);


# train feed-forward model using mini-batch gradient descent
optimiser = {'name': 'gradient_descent', 'mini_batch_sz': 32}
lambd = 1e-4
params, grads, loss_log = model_fit(X_train, Y_train, n_h=[8, 3], optimiser=optimiser, epochs=10000, learning_rate=lambd)

# evaluate model
Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1), "Optimiser: mini-batch gradient descent")

# plot loss
plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.title("Learning rate = {}".format(lambd))
plt.show()


# train feed-forward model using momentum
optimiser = {'name': 'momentum', 'mini_batch_sz': 32, 'beta': 0.9}
lambd = 1e-4
params, grads, loss_log = model_fit(X_train, Y_train, n_h=[8, 3], optimiser=optimiser, epochs=10000, learning_rate=lambd)

# evaluate model
Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1), "Optimiser: momentum")

# plot loss
plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.title("Learning rate = {}".format(lambd))
plt.show()

# train feed-forward model using adam
optimiser = {'name': 'adam', 'mini_batch_sz': 32, 'beta1': 0.9, 'beta2': 0.999, 'epsilon':1e-8}
lambd = 1e-4
params, grads, loss_log = model_fit(X_train, Y_train, n_h=[8, 3], optimiser=optimiser, epochs=10000, learning_rate=lambd)

# evaluate model
Y_hat_train = u.model_predict(params, X_train)
Y_hat_test = u.model_predict(params, X_test)
train_acc = 100 * (1 - np.mean(np.abs(Y_hat_train - Y_train)))
test_acc = 100 * (1 - np.mean(np.abs(Y_hat_test - Y_test)))
print("{:.1f}% training acc.".format(train_acc))
print("{:.1f}% test acc.".format(test_acc))

u.plot_model(lambda x: u.model_predict(params, x), X_test, Y_test.reshape(-1), "Optimiser: Adam")

# plot loss
plt.plot(loss_log)
plt.ylabel('loss')
plt.xlabel('epochs (x100)')
plt.title("Learning rate = {}".format(lambd))
plt.show()
