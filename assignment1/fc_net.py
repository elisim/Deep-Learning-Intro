from layers import *
import numpy as np

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a sigmoid as last layer. This will also implement
    dropout and batch normalization as options.
    """
    
    def __init__(self, layer_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - layer_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: a boolean defines whether or not use batch norm
        """
        self.params = initialize_parameters([input_dim] + layer_dims + [num_classes])
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.num_layers = 1 + len(layer_dims)
        
        
def initialize_parameters(layer_dims):
    """
    input: 
        an array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output sigmoid)
    output: 
        a dictionary containing the initialized W and b parameters of each layer (W1...WL, b1...bL).
    Hint: 
        Use the randn and zeros functions of numpy to initialize W and b, respectively
    """
    params = {}
    layer_input_dim = layer_dims[0]
    num_classes = layer_dims[-1]
    
    # input-> hidden_layer_1 -> hidden_layer_2 -> ... -> hidden_layer_last
    for idx, dim in enumerate(layer_dims[1:-1]): # enumrate all hidden layers
        layer_num = str(idx+1)
        params['W' + layer_num] = np.random.randn(layer_input_dim, dim)
        params['b' + layer_num] = np.zeros(dim)
        layer_input_dim = dim    
    
    # hidden_layer_last -> output
    num_layers = len(layer_dims) - 1
    params['W' + str(num_layers)] = np.random.randn(layer_input_dim, num_classes)
    params['b' + str(num_layers)] = np.zeros(num_classes)
    
    return params
    