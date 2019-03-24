import numpy as np


def softmax(Z):
    """
    Compute softmax values for each sets of scores in x
    """
    activiation_cache = Z
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z, axis=1, keepdims=True), activiation_cache # sum row-wise


def softmax_backward(dA, activation_cache):
    """
    Description:
        Implements backward propagation for a softmax unit
    Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
    Output:
        dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache
    # e_Z = np.exp(Z - np.max(Z))
    # #TODO: fix the formula
    # softmax_val = e_Z / e_Z.sum(axis=0)
    # dZ = dA * softmax_val * (1-softmax_val)
    dZ = dA - 1
    return dZ


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
    n_activations = A.shape[0]
    # TODO: check it that ok that i added the W.T here
    Z = np.dot(A.reshape(n_activations, -1), W) + b
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
    cache = {'linear_cache': linear_cache, 'activation_cache': activation_cache}
    return A, cache


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
    linear_cache, activation_cache = cache['linear_cache'], cache['activation_cache']
    activation_backward = globals()[activation + '_backward']
    dZ = activation_backward(dA, activation_cache)
    return linear_backward(dZ, linear_cache)


# def sigmoid_backward(dA, activation_cache):
#     """
#     Description:
#         Implements backward propagation for a sigmoid unit
#     Input:
#         dA – the post-activation gradient
#         activation_cache – contains Z (stored during the forward propagation)
#     Output:
#         dZ – gradient of the cost with respect to Z
#     """
#     # dZ = sig(Z)*(1-sig(Z))
#     Z = activation_cache
#     sig_Z = sigmoid(Z)
#     d_sig = sig_Z * (1 - sig_Z)
#     dZ = d_sig * dA
#     return dZ
#

# def sigmoid(Z):
#     """
#     Numerically-stable sigmoid function
#     sigmoid(x) = 1/(1+exp(−x)) = exp(x)/(exp(x)+1)
#
#     Input:
#         Z – the linear component of the activation function
#     Output:
#         A – the activations of the layer
#         activation_cache – returns Z, which will be useful for the backpropagation
#     """
#     if Z >= 0:
#         e = np.exp(-Z)
#         A = 1 / (1 + e) # prevent division by zero if Z -> +inf
#     else:
#         e = np.exp(Z)
#         A = e / (1 + e) # prevent division by zero if Z -> -inf
#     activation_cache = Z
#     return A, activation_cache
