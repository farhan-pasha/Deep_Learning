# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import utils1 as u

%matplotlib inline

def load_dataset():
    """
    create dataset
    
    Returns:
    (X_train, Y_train), (X_test, Y_test), classes -- training and test data
    """
    
    np.random.seed(2019) # for reproducibility
    n = 256
    X = np.zeros((2*n,2))
    Y = np.zeros((2*n,1), dtype='uint8')
    classes = ['red', 'blue']

    for k in range(2):
        i = range(k*n, (k+1)*n)
        Y[i, 0] = k
        # spirograph data points
        theta = np.linspace(k*np.pi, 2*(k+1)*np.pi, n)
        rho = np.sin(3*(theta+k*np.pi)) + np.random.randn(n) * 0.1
        X[i] = np.c_[rho*np.sin(theta), rho*np.cos(theta)]
    
    
    index = np.random.choice(2*n, size=int(0.05 * 2*n), replace=False)    
    Y_s = Y[index, 0]
    np.random.shuffle(Y_s)
    Y[index, 0] = Y_s
    
    X_train = np.concatenate((X[0:int(n/2),:], X[n:3*int(n/2),:]))
    X_test = np.concatenate((X[int(n/2):n,:], X[3*int(n/2):2*n,:]))
    Y_train = np.concatenate((Y[0:int(n/2),:], Y[n:3*int(n/2),:]))
    Y_test = np.concatenate((Y[int(n/2):n,:], Y[3*int(n/2):2*n,:]))
                             
    return (X_train, Y_train), (X_test, Y_test), classes

(X_train, Y_train), (X_test, Y_test), classes = load_dataset()

n_train = X_train.shape[0]
n_test = X_test.shape[0]
feature_count = X_train.shape[1]

print("training set: {} images".format(n_train))
print("test set: {} images".format(n_test))
print("features per sample: {} scalars".format(feature_count))
plt.scatter(X_test[:,0], X_test[:,1], c=Y_test.reshape(-1), cmap=plt.cm.Spectral);
print(X_train.shape)

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

# Define and train Logistic regression model
model = sklearn.linear_model.LogisticRegressionCV(cv=3);
model.fit(X_train, Y_train.ravel());
# Visualise class regions and decision boundaries
plot_model(lambda x: model.predict(x), X_test, Y_test.reshape(-1))
plt.title("Logistic Regression (test data)")

# Report model performance on training and test data
Y_hat_train = model.predict(X_train).reshape(Y_train.shape)
Y_hat_test = model.predict(X_test).reshape(Y_test.shape)
n_train = Y_train.shape[0]
n_test = Y_test.shape[0]

#print(Y_hat_train-Y_train)
#print(Y_train-Y_hat_train)

loss=np.abs(Y_train-Y_hat_train)
train_acc = loss.sum()*100/Y_hat_train.shape[0]
loss=np.absolute(Y_test-Y_hat_test)
test_acc =  loss.sum()*100/Y_hat_test.shape[0]

print("model accuracy (training) = {:.1f}%".format(train_acc))
print("testing model (test)= {:.1f}%".format(test_acc))

def model_config(X, Y, hidden_units=3):
    """
    Arguments:
    X -- n data samples, shape = (n, n_x)
    Y -- ground truth label, column vector of shape (n, 1)
    
    Returns:
    n_x -- number of units in the input layer
    n_y -- number of units in the output layer
    n_h -- number of units in the hidden layer
    """
    

    n_x = X.shape[1]
    n_h = hidden_units
    n_y = Y.shape[1]
    
    assert(X.shape[0] == Y.shape[0])
    return (n_x, n_y, n_h)

X, Y, hidden_units = u.model_config_test()
(n_x, n_y, n_h) = model_config(X, Y, hidden_units)

print("{} input units".format(n_x))
print("{} output unit".format(n_y))
print("{} hidden units".format(n_h))

# Model parameter initialisation
def init_model_parameters(n_x, n_y, n_h):
    """
    n_x -- number of units in the input layer
    n_y -- number of units in the output layer
    n_h -- number of units in the hidden layer
    
    Returns: dictionary containing your parameters:
        W1 -- initialised weight matrix of shape (n_x, n_h)
        b1 -- initialised bias vector of shape (1, n_h)
        W2 -- initialised weight matrix of shape (n_h, n_y)
        b2 -- initialised bias vector of shape (1, n_y)
    """
    
    W1 = np.random.randn(n_x,n_h)*.01
    b1 = np.zeros((1,n_h))
    W2 = np.random.randn(n_h,n_y)*.01
    b2 = np.zeros((1,n_y))

    assert(W1.shape == (n_x, n_h))
    assert(b1.shape == (1, n_h))
    assert(W2.shape == (n_h, n_y))
    assert(b2.shape == (1, n_y))
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

np.random.seed(2019)
params = init_model_parameters(n_x=2, n_y=1, n_h=3)
np.set_printoptions(precision=5, suppress=True)
print("W1 = {}".format(params["W1"]))
print("b1 = {}".format(params["b1"]))
print("W2.T = {}".format(params["W2"].T))
print("b2 = {}".format(params["b2"]))

# Forward propagation (inference)
def forward_prop(params, X, Y=None):
    """
    Compute the layer activations and loss if needed

    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h)
        b1 -- initialised bias vector of shape (1, n_h)
        W2 -- initialised weight matrix of shape (n_h, n_y)
        b2 -- initialised bias vector of shape (1, n_y)
    X -- n data samples, shape = (n, n_x)
    Y -- optional argument, ground truth label, column vector of shape (n, 1)

    Returns:
    loss -- cross-entropy loss or NaN if Y=None
    cache -- a dictionary containing "Z1", "A1", "Z2", A2"
        Z1 -- matrix of shape (n, n_h)
        A1 -- matrix of shape (n, n_h)
        Z2 -- matrix of shape (n, n_y)
        A2 -- matrix of shape (n, n_y)
    """
    
    
    n = X.shape[0]
    
    # Retrieve model parameters
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    # Forward prop: compute cache from X
    Z1 = X@W1+b1
    A1 = np.tanh(Z1)
    Z2 = A1@W2+b2
    A2 = u.sigmoid(Z2)
    
    n_x = params["W1"].shape[0]
    n_y = params["W2"].shape[1]
    n_h = params["W1"].shape[1]
    assert(A1.shape == (n, n_h))
    assert(Z1.shape == (n, n_h))
    assert(A2.shape == (n, n_y))
    assert(Z2.shape == (n, n_y))
    
    loss = float('nan')
    if Y is not None:
        Y_hat = A2
        # Compute the cross-entropy loss
        loss = -1/n * (Y.T@np.log(Y_hat)+(1-Y).T@np.log(1-Y_hat))

        loss = np.squeeze(loss)
        assert(loss.dtype == float)
        
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}, loss


params, X, Y = u.forward_prop_test()
cache, loss = forward_prop(params, X, Y)

print("np.mean(Z1) = {:.5f}".format(np.mean(cache["Z1"])))
print("np.mean(A1) = {:.5f}".format(np.mean(cache["A1"])))
print("np.mean(Z2) = {:.5f}".format(np.mean(cache["Z2"])))
print("np.mean(A2) = {:.5f}".format(np.mean(cache["A2"])))
print("loss = {:.5f}".format(loss))

# Backward_propagation
def back_prop(params, X, Y, cache):
    """
    Compute back-propagation gradients
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h)
        b1 -- initialised bias vector of shape (1, n_h)
        W2 -- initialised weight matrix of shape (n_h, n_y)
        b2 -- initialised bias vector of shape (1, n_y)
    X -- n data samples, shape = (n, n_x)
    Y -- ground truth label, column vector of shape (n, 1)
    cache -- dictionary containing "Z1", "A1", "Z2", A2"
        Z1 -- matrix of shape (n, n_h)
        A1 -- matrix of shape (n, n_h)
        Z2 -- matrix of shape (n, n_y)
        A2 -- matrix of shape (n, n_y)
    
    Returns:
    grads -- dictionary containing your gradients with respect to all parameters
        dW1 -- weight gradient matrix of shape (n_x, n_h)
        db1 -- bias gradient vector of shape (1, n_h)
        dW2 -- weight gradient matrix of shape (n_h, n_y)
        db2 -- bias gradient vector of shape (1, n_y)
    """
    
    n = X.shape[0]
    
    # Retrieve w1 and w2 weights from params dictionary
    W1 = params["W1"]
    W2 = params["W2"]
        
    # Retrieve A1 and A2 from cache dictionary
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backprop calculation for dw1, db1, dw2, db2
    dZ2 = A2-Y
    dW2 = 1/n*(A1.T @ dZ2)
    db2 = 1/n*sum(dZ2)
    print(dZ2)
    print("here")
    print(W2.T)
    print("here")
    print(A1)
    dZ1 = (dZ2 @ W2.T)*(1-(A1**2))
    dW1 = 1/n*(X.T @ dZ1)
    db1 = 1/n*sum(dZ1)
    ### END OF YOUR CODE SEGMENT ###  
        
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}



params, X, Y, cache = u.back_prop_test()
grads = back_prop(params, X, Y, cache)

print("dW1 = {}".format(grads["dW1"]))
print("db1 = {}".format(grads["db1"]))
print("dW2.T = {}".format(grads["dW2"].T))
print("db2 = {}".format(grads["db2"]))

# Update model parameters
def update_params(params, grads, learning_rate=0.8):
    """
    Updates parameters using the gradient descent
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- weight matrix of shape (n_x, n_h)
        b1 -- bias vector of shape (1, n_h)
        W2 -- weight matrix of shape (n_h, n_y)
        b2 -- bias vector of shape (1, n_y)
    grads -- dictionary containing gradients
        dW1 -- weight gradient matrix of shape (n_x, n_h)
        db1 -- bias gradient vector of shape (1, n_h)
        dW2 -- weight gradient matrix of shape (n_h, n_y)
        db2 -- bias gradient vector of shape (1, n_y)
    learning_rate -- learning rate of the gradient descent (hyperparameter)

    Returns:
    params -- dictionary containing updated parameters
        W1 -- updated weight matrix of shape (n_x, n_h)
        b1 -- updated bias vector of shape (1, n_h)
        W2 -- updated weight matrix of shape (n_h, n_y)
        b2 -- updated bias vector of shape (1, n_y)
    """

    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1- learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    ### END CODE HERE ###
        
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

params, grads = u.update_params_test()
params = update_params(params, grads)

print("W1 = {}".format(params["W1"]))
print("b1 = {}".format(params["b1"]))
print("W2.T = {}".format(params["W2"].T))
print("b2 = {}".format(params["b2"]))

# Parameter optimisation using backprop
def model_fit(params, X, Y, epochs=2000, learning_rate=0.8, verbose=False):
    """
    Optimise model parameters performing gradient descent
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- initialised weight matrix of shape (n_x, n_h)
        b1 -- initialised bias vector of shape (1, n_h)
        W2 -- initialised weight matrix of shape (n_h, n_y)
        b2 -- initialised bias vector of shape (1, n_y)
    X -- n data samples  (n, n_x)
    Y -- groud truth label vector of size (n, n_y)
    epochs -- number of iteration updates through dataset
    learning_rate -- learning rate of the gradient descent
    
    Returns:
    params -- dictionary with optimised parameters
    grads -- dictionary with final gradients
    loss_log -- list of loss values for every 1000 updates
    """
    
    loss_log = []
    for i in range(epochs):
        cache, loss = forward_prop(params,X,Y)
        grads = back_prop(params, X, Y, cache)
        params = update_params(params, grads)
        
        # logs
        if i % 1000 == 0:
            loss_log.append(np.asscalar(loss))
            if verbose:
                print("Loss after {} iterations: {:.3f}".format(i, loss))
     
    return params, grads, loss_log

params, X, Y = u.forward_prop_test()
params, grads, loss_log = model_fit(params, X, Y, epochs = 2100, verbose=True)

print("W1 = {}".format(params["W1"]))
print("b1 = {}".format(params["b1"]))
print("W2.T = {}".format(params["W2"].T))
print("b2 = {}".format(params["b2"]))


# Model inference
def model_predict(params, X):
    '''
    Predict class label using model parameters
    
    Arguments:
    params -- dictionary containing model parameters
        W1 -- optimised weight matrix of shape (n_x, n_h)
        b1 -- optimised bias vector of shape (1, n_h)
        W2 -- optimised weight matrix of shape (n_h, n_y)
        b2 -- optimised bias vector of shape (1, n_y)
    X -- n data samples  (n, n_x)
    
    Returns:
    Y_hat -- vector with class predictions for examples in X
    '''
    
    cache, _ = forward_prop(params, X, )
    Y_hat = (cache["A2"] > 0.5).astype(int) # Convert activations to {0,1} predictions
    
    n = X.shape[0]
    assert(Y_hat.shape == (n, 1))    
    return Y_hat


params, X = u.model_predict_test()
Y_hat = model_predict(params, X)

print("predictions.T = {}".format(Y_hat.T))


# SLFN model
def slfn_model(X_train, Y_train, X_test, Y_test, hidden_units=3, epochs=10000, learning_rate=0.5):
    '''
    Build, train and evalaute the logistic regression model
    
    Arguments:
    X_train -- training set a numpy array of shape (n_train, n_x)
    Y_train -- training groud truth vector (0=dog, 1=cat) of size (n_train, n_y)
    X_test -- testing set a numpy array of shape (n_test, n_x)
    Y_test -- testing groud truth vector (0=dog, 1=cat) of size (n_test, n_y)
    hidden_units -- number of units in hidden layer
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
    
    (n_x, n_y, n_h) = model_config(X_train, Y_train, hidden_units)
    params = init_model_parameters(n_x, n_y, n_h)
    params, grads, loss = model_fit(params, X_train, Y_train, epochs, learning_rate,verbose=True)
    Y_hat_train = model_predict(params, X_train)
    Y_hat_test = model_predict(params, X_test)
    train_acc = (100 * (1 - np.mean(np.abs(Y_train-Y_hat_train))))
    test_acc = (100 * (1 - np.mean(np.abs(Y_test - Y_hat_test))))

    print("{:.1f}% training acc.".format(train_acc))
    print("{:.1f}% test acc.".format(test_acc))
        
    return {"PARAMS": params, "LOSS": loss, "GRADS": grads, "ACC": [train_acc, test_acc], "LR": learning_rate}

model = slfn_model(X_train, Y_train, X_test, Y_test)
params = model["PARAMS"]

# Plot the decision boundary
plot_model(lambda x: model_predict(params, x), X_test, Y_test.reshape(-1))
_ = plt.title("SLFN with {} hidden units".format(params["W1"].shape[1]))



plt.figure(figsize=(16, 32))
for i, hidden_units in enumerate([1, 2, 3, 4, 8, 16]):
    print("SLFN with {} hidden units".format(hidden_units))
    model = slfn_model(X_train, Y_train, X_test, Y_test, hidden_units)
    params = model["PARAMS"]
    plt.subplot(5, 2, i+1)
    plot_model(lambda x: model_predict(params, x), X_test, Y_test.reshape(-1))
    _ = plt.title("SLFN with {} hidden units".format(hidden_units))

