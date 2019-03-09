import numpy as np

def linear_forward(A, W, b):
    """
    Description: Implement the linear part of a layer's forward propagation.
    
    input:
        A – the activations of the previous layer
        W – the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
        B – the bias vector of the current layer (of shape [size of current layer, 1])

    Output:
        Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
        linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)
    """ 
    n_activations = A.shape[0]
    Z = np.dot(A.reshape(n_activations,-1), W) + b 
    linear_cache = (A, W, b)
    return Z, linear_cache


def sigmoid(Z):
    """
    Numerically-stable sigmoid function
    sigmoid(x) = 1/(1+exp(−x)) = exp(x)/(exp(x)+1)
    
    Input:
        Z – the linear component of the activation function
    Output:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation
    """
    if Z >= 0:
        e = np.exp(-Z)
        A = 1 / (1 + e) # prevet division by zero if Z -> +inf
    else:
        e = np.exp(Z)
        A = e / (1 + e) # prevet division by zero if Z -> -inf
    activation_cache = Z
    return A, activation_cache


def relu(Z):
    """
    Input:
        Z – the linear component of the activation function
    Output:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = Z * (Z >= 0)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    Description:
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    Input:
        A_prev – activations of the previous layer
        W – the weights matrix of the current layer
        B – the bias vector of the current layer
        Activation – the activation function to be used (a string, either “sigmoid” or “relu”)
    Output:
        A – the activations of the current layer
        cache – a joint dictionary containing both linear_cache and activation_cache
    """
    act = globals()[activation] # get activation function 
    Z, linear_cache = linear_forward(A_prev, W, B)
    A, activation_cache = act(Z)
    cache = {
        'linear_cache': linear_cache,
        'activation_cache': activation_cache
    }
    return A, cache
        