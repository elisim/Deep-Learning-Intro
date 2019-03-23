import numpy as np

def softmax(Z):
    """
    Compute softmax values for each sets of scores in x
    """
    activiation_cache = Z
    e_Z = np.exp(Z - np.max(Z))
    return e_Z.T / np.sum(e_Z, axis=1), activiation_cache # sum row-wise


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
    e_Z = np.exp(Z - np.max(Z))
    #TODO: fix the formula
    softmax_val = e_Z / e_Z.sum(axis=0)
    dZ = dA * softmax_val * (1-softmax_val)
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
