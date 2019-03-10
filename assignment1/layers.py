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


def Linear_backward(dZ, cache):
    """
    Description:
        Implements the linear part of the backward propagation process for a single layer
    Input:
        dZ – the gradient of the cost with respect to the linear output of the current layer (layer l)
        cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Output:
        dA_prev - Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW - Gradient of the cost with respect to W (current layer l), same shape as W
        db - Gradient of the cost with respect to b (current layer l), same shape as b
    """
    pass


def linear_activation_backward(dA, cache, activation):
    """
    Description:
        Implements the backward propagation for the LINEAR->ACTIVATION layer. The function
        first computes dZ and then applies the linear_backward function.
    Input:
        dA – post activation gradient of the current layer
        cache – contains both the linear cache and the activations cache
    Output:
        dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW – Gradient of the cost with respect to W (current layer l), same shape as W
        db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    pass


def relu_backward(dA, activation_cache):
    """
    Description:
        Implements backward propagation for a ReLU unit
    Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
    Output:
        dZ – gradient of the cost with respect to Z
    """
    pass


def sigmoid_backward(dA, activation_cache):
    """
    Description:
        Implements backward propagation for a sigmoid unit
    Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
    Output:
        dZ – gradient of the cost with respect to Z
    """
    pass