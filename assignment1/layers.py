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
    return Z, activation_cache
    