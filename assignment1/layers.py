import numpy as np


def softmax(Z):
    """
    Compute softmax values for each sets of scores in x
    """
    activiation_cache = Z
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z, axis=1, keepdims=True), activiation_cache # sum row-wise


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
    # dZ = 1 for Z > 0 and 0 otherwise
    Z = activation_cache
    d_relu = (Z > 0) * 1
    dZ = d_relu * dA
    return dZ


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
    Z = np.dot(A, W) + b
    linear_cache = {'A': A, 'W': W, 'b': b}
    return Z, linear_cache


def linear_backward(dZ, cache):
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
    # f = WA+b
    # dA = W', dw = A', db = 1
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    N = A_prev.shape[0]
    A_prev_reshaped = A_prev.reshape(N, -1)

    dA_prev = dZ.dot(W.T).reshape(A_prev.shape)
    dW = A_prev_reshaped.T.dot(dZ) / N
    db = np.sum(dZ, axis=0, keepdims=True) / N

    return dA_prev, dW, db


def linear_activation_forward(A_prev, W, B, activation, use_batchnorm):
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
    batchnorm_cache = ()
    act = globals()[activation] # get activation function
    Z, linear_cache = linear_forward(A_prev, W, B)
    if use_batchnorm:
        Z, batchnorm_cache = apply_batchnorm(Z)
    A, activation_cache = act(Z)
    cache = {'linear_cache': linear_cache, 'activation_cache': activation_cache}
    return A, cache, batchnorm_cache


def linear_activation_backward(dA, cache, activation, use_batchnorm, batchnorm_cache):
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
    linear_cache, activation_cache = cache['linear_cache'], cache['activation_cache']
    activation_backward = globals()[activation + '_backward']
    dZ = activation_backward(dA, activation_cache)
    if use_batchnorm:
        dZ = batchnorm_backward(dZ, batchnorm_cache)
    return linear_backward(dZ, linear_cache)

def batchnorm_backward(dZ_norm, batchnorm_cache):
    activation = batchnorm_cache['activation']
    miu = batchnorm_cache['miu']
    var = batchnorm_cache['var']

    N = activation.shape[0]

    dvar = np.sum(dZ_norm*(activation-miu), axis=0)*(-1.0/2)*((var + 1e-8)**(-3.0/2))
    dmiu = np.sum(dZ_norm*(-1.0/np.sqrt(var + 1e-8)), axis=0) + dvar*(1.0/N)*np.sum(2.0*(activation-miu), axis=0)

    dZ = dZ_norm * (1.0/np.sqrt(var + 1e-8)) + dmiu*(1.0/N) + dvar*(2.0/N)*(activation - miu)

    return dZ

def apply_batchnorm(activation):
    """
    Description:
    performs batchnorm on the received activation values of a given layer.
    Input:
    A - the activation values of a given layer
    output:
    NA - the normalized activation values, based on the formula learned in class
    batchnorm_cache - cache with the info of the batchnorm forward for backpropogation
    """
    epsilon =  1e-8
    miu = np.mean(activation, axis=0)
    var = np.var(activation, axis=0)
    NA = (activation - miu) / np.sqrt(var + epsilon)
    batchnorm_cache = {'activation': activation, 'activation_norm': NA, 'miu': miu,'var': var}

    return NA, batchnorm_cache


def dropout_forward(x, p):
    """
    Performs the forward pass for (inverted) dropout.

    :param x: Input data, of any shape
    :param p: Dropout parameter. We keep each neuron output with probability p.
    :return:
        - out: Array of the same shape as x.
        - cache: dropout mask that was used to multiply the input

    """
    mask = (np.random.rand(*x.shape) < p) / p  # inverted dropout mask
    out = x * mask  # drop!
    return out, mask


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    :param dout: Upstream derivatives, of any shape
    :param cache: mask from dropout_forward
    :return:
    """
    mask = cache
    dx = dout * mask
    return dx

