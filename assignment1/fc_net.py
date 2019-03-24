from layers import *
import numpy as np
from sklearn.metrics import accuracy_score


def initialize_parameters(layer_dims):
    """
    input:
        an array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output sigmoid)
    output:
        a dictionary containing the initialized W and b parameters of each layer (W1...WL, b1...bL).
    """
    params = {}
    layer_input_dim = layer_dims[0]
    num_classes = layer_dims[-1]

    # input-> hidden_layer_1 -> hidden_layer_2 -> ... -> hidden_layer_last
    for idx, dim in enumerate(layer_dims[1:]): # enumrate all hidden layers
        layer_num = str(idx+1)
        params['W' + layer_num] = np.random.randn(layer_input_dim, dim) * 0.1
        params['b' + layer_num] = np.zeros(dim)
        layer_input_dim = dim

    # TODO: is below useless with softmax?
    # hidden_layer_last -> output
    num_layers = len(layer_dims)
    params['W' + str(num_layers)] = np.random.randn(layer_input_dim, num_classes)
    params['b' + str(num_layers)] = np.zeros(num_classes)

    return params


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


def L_model_forward(X, parameters, use_batchnorm):
    """
    forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation

    :param X: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation
    :return: (the last post-activation value , a list of all the cache objects)
    """

    layer_input = X
    caches = []
    num_layers = len([key for key in parameters.keys() if key.startswith('W')])

    for layer_idx in range(1, num_layers):
        W, b = parameters['W' + str(layer_idx)], parameters['b' + str(layer_idx)]
        layer_input, layer_cache = linear_activation_forward(layer_input, W, b, 'relu')
        caches.append(layer_cache)
        if use_batchnorm:
            layer_input = apply_batchnorm(layer_input)

    # last layer
    W, b = parameters['W' + str(num_layers)], parameters['b' + str(num_layers)]
    last_post_activation, layer_cache = linear_activation_forward(layer_input, W, b, 'softmax')
    caches.append(layer_cache)

    return last_post_activation, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss.

    :param AL: – probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return: the cross-entropy cost
    """

    #TODO: check what happen when AL got invalid value for log
    return - np.sum((Y * np.log(AL))) / Y.shape[0]
    #return -np.sum((Y * np.log(AL)) + ((1-Y) * np.log(1-AL))) / Y.shape[0]


def apply_batchnorm(activation):
    epsilon = 0.000001
    miu = np.sum(activation) / activation.shape[0]
    sigma = (np.sum(activation - miu) ** 2) / activation.shape[0]
    return (activation - miu) / (sigma + epsilon) ** 0.5


def L_model_backward(AL, Y, caches):
    """
    Backward propagation process for the entire network.

    :param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y: the true labels vector (the "ground truth" - true classifications)
    :param caches: list of caches containing for each layer: a) the linear cache; b) the activation cache
    :return: a dictionary with the gradients
    """

    grads = {}
    num_layers = len(caches)

    # dL / dA = -(Y/A) + ((1-Y)/1-A)
    #TODO: fix parameters for softmax_backward (what should be dA)
    #last_layer_dA = -((Y / AL) - ((1-Y)/(1-AL)))
    last_layer_idx = num_layers

    dA, dW, db = linear_backward(AL - Y, caches[-1]['linear_cache'])
    grads['dA' + str(last_layer_idx)] = dA
    grads['dW' + str(last_layer_idx)] = dW
    grads['db' + str(last_layer_idx)] = db

    for layer_idx in reversed(range(1, num_layers)):
        dA, dW, db = linear_activation_backward(dA , caches[layer_idx - 1], "relu")
        grads['dA' + str(layer_idx)] = dA
        grads['dW' + str(layer_idx)] = dW
        grads['db' + str(layer_idx)] = db

    return grads


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


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent

    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :param grads: a python dictionary containing the gradients (generated by L_model_backward)
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :return: – the updated values of the parameters object provided as input
    """

    num_layers = len([key for key in parameters.keys() if key.startswith('W')])

    for layer_idx in range(1, num_layers + 1):
        old_W, dW = parameters['W' + str(layer_idx)], grads['dW' + str(layer_idx)]
        old_b, db = parameters['b' + str(layer_idx)], grads['db' + str(layer_idx)]

        parameters['W' + str(layer_idx)] = old_W - learning_rate * dW
        parameters['b' + str(layer_idx)] = old_b - learning_rate * db

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """

    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: the learning rate
    :param num_iterations: number of iterations
    :param batch_size: the number of examples in a single training batch.
    :return: (parameters, costs) - the parameters learnt by the system during the training (the same parameters
                                    that were updated in the update_parameters function) and the values of the cost
                                    function (calculated by the compute_cost function). One value is to be saved
                                    after each 100 training iterations (e.g. 3000 iterations -> 30 values)..
    """
    # initialization
    parameters = initialize_parameters([X.shape[1]] + layers_dims)
    costs = []

    for i in range(num_iterations):
        for X_batch, Y_batch in next_batch(X, Y, batch_size):
            # choose the batch
            # batch_idx = np.random.choice(np.arange(X.shape[1]), size=batch_size, replace=False)
            # X_batch, Y_batch = X[:, batch_idx], Y[batch_idx]

            # forward pass
            AL, caches = L_model_forward(X_batch, parameters, False)

            # compute the cost and document it
            cost = compute_cost(AL, Y_batch)
            print(cost)

            # backward pass
            grads = L_model_backward(AL, Y_batch, caches)

            # update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1 == 0:
            costs.append(cost)

    return parameters, costs


def next_batch(X, y, batchSize):
    # loop over our dataset X in mini-batches of size batchSize
    for i in np.arange(0, X.shape[1], batchSize):
        # yield a tuple of the current batched data and labels
        yield (X[i: i+batchSize, :], y[i: i+batchSize, :])


def predict(X, Y, parameters):
    """
    Description:
        The function receives an input data and the true labels and calculates the accuracy of
        the trained neural network on the data.
    Input:
        X – the input data, a numpy array of shape (height*width, number_of_examples)
        Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
        parameters – a python dictionary containing the DNN architecture’s parameters
    Output:
        accuracy – the accuracy measure of the neural net on the provided data (i.e. the
        percentage of the samples for which the correct label receives over 50% of the
        confidence score). Use the softmax function to normalize the output values.
    """
    # scores: Array of shape (num_classes, number_of_examples) giving classification scores,
    # where scores[c, i] is the classification score for X[i] and class c.
    scores, caches = L_model_forward(X, parameters, use_batchnorm=False) # TODO: ask Gilad where use_batchnorm should come from.
    predictions = np.argmax(scores, axis=0)
    # TODO: check if none of the classes is above 50%? 0.1 for all classes for example
    return accuracy_score(Y ,predictions)

