import keras
import keras.backend as K
import keras.layers as KL
from keras.regularizers import l2

IMAGES_DIM = 250


def build_hani(**model_params):
    """
    :return: the network the mentioned in the Hani et el. paper:
    --------------------------------------------------------
    Khalil-Hani, M., & Sung, L. S. (2014). A convolutional neural
    network approach for face verification. High Performance Computing
    & Simulation (HPCS), 2014 International Conference on, (3), 707–714.
    doi:10.1109/HPCSim.2014.6903759
    """
    def tanh_scaled(x):
        A = 1.7159
        B = 2 / 3
        return A * K.tanh(B * x)

    act = model_params.get('act', tanh_scaled)
    dropout = model_params.get('dropout', 0)
    batchnorm = model_params.get('batchnorm', False)
    loss = model_params.get('loss', contrastive_loss)
    learning_rate = model_params.get('learning_rate', 1e-3)
    input_shape = (IMAGES_DIM, IMAGES_DIM, 1)
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)

    model = keras.Sequential()
    initialize_weights_conv = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84)  # filters initialize
    initialize_weights_dense = keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=84)  # dense initialize
    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

    model.add(KL.Conv2D(5, (6, 6), strides=(2, 2), activation=act, input_shape=input_shape,
                        kernel_initializer=initialize_weights_conv, kernel_regularizer=l2(1e-2)))
    if batchnorm:
        model.add(KL.BatchNormalization())
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(14, (6, 6), strides=(2, 2), activation=act, kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
    if batchnorm:
        model.add(KL.BatchNormalization())
    model.add(KL.MaxPool2D())

    model.add(KL.Dropout(dropout))
    model.add(KL.Conv2D(60, (6, 6), activation=act, kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
    if batchnorm:
        model.add(KL.BatchNormalization())
    model.add(KL.MaxPool2D())

    model.add(KL.Flatten())

    model.add(KL.Dense(40, activation=act, kernel_regularizer=l2(1e-4),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))
    model.add(KL.Dense(40, activation=None, kernel_regularizer=l2(1e-4),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(first_input)
    encoded_r = model(second_input)

    # calculate similarity
    if loss == 'binary_crossentropy':
        L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        similarity = KL.Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
    else:
        similarity = KL.Lambda(euclidean_distance)([encoded_l, encoded_r])

    
    # final network
    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    print(loss)
    final_network.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return final_network

def build_hani_best_model(**model_params):
    """
    :return: the network the mentioned in the Hani et el. paper: 
    --------------------------------------------------------
    Khalil-Hani, M., & Sung, L. S. (2014). A convolutional neural
    network approach for face verification. High Performance Computing
    & Simulation (HPCS), 2014 International Conference on, (3), 707–714.
    doi:10.1109/HPCSim.2014.6903759
    but with optimized hyperparmeters after the hyperas exection
    """

    act = model_params.get('act', 'relu')
    dropout = model_params.get('dropout', 0)
    batchnorm = model_params.get('batchnorm', False)
    loss = model_params.get('loss', contrastive_loss)
    learning_rate = model_params.get('learning_rate', 1e-3)
    input_shape = (IMAGES_DIM, IMAGES_DIM, 1)
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)

    model = keras.Sequential()
    initialize_weights_conv = keras.initializers.glorot_uniform(seed=84)  # filters initialize
    initialize_weights_dense = keras.initializers.glorot_uniform(seed=84)  # dense initialize  
    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

    model.add(KL.Conv2D(5, (6, 6), strides=(2, 2), activation=act, input_shape=input_shape,
                        kernel_initializer=initialize_weights_conv, kernel_regularizer=l2(0.03148394777069553)))

    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.3065491917788273))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(14, (6, 6), strides=(2, 2), activation=act, kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(0.054048669207277224)))
    #model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.4797699256757003))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(60, (6, 6), activation=act, kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(0.06189584230948173)))

    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.020012398358003752))
    model.add(KL.MaxPool2D())

    model.add(KL.Flatten())

    model.add(KL.Dense(40, activation=act, kernel_regularizer=l2(0.082430594544267),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))
    model.add(KL.Dropout(0.012533877486030926))
    model.add(KL.Dense(40, activation=None, kernel_regularizer=l2(0.046085917780636185),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))
    model.add(KL.Dropout(0.05086327591390307))
        
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(first_input)
    encoded_r = model(second_input)

    # calculate similarity
    L1_distance = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_l, encoded_r])
    similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias)(L1_distance)
    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)                                                                         
    optimizer = keras.optimizers.SGD(lr=0.03863427079945416, momentum=0.8962431889503087, decay=0.019965108317109886)      
    final_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return final_network


def build_paper_network(**model_params):
    """
    :return: the network the mentioned in the original paper:
    --------------------------------------------------------
    Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov.
    "Siamese neural networks for one-shot image recognition."
    In ICML deep learning workshop, vol. 2. 2015.
    """
    filter_size_conv1 = model_params.get('filter_size_conv1', 10)
    filter_size_conv2 = model_params.get('filter_size_conv2', 7)
    filter_size_conv3 = model_params.get('filter_size_conv3', 4)
    filter_size_conv4 = model_params.get('filter_size_conv4', 4)
    n_filters_conv1 = model_params.get('n_filters_conv1', 64)
    n_filters_conv2 = model_params.get('n_filters_conv2', 128)
    n_filters_conv3 = model_params.get('n_filters_conv3', 128)
    n_filters_conv4 = model_params.get('n_filters_conv4', 256)
    l2_conv1 = model_params.get('l2_conv1', 1e-2)
    l2_conv2 = model_params.get('l2_conv2', 1e-2)
    l2_conv3 = model_params.get('l2_conv3', 1e-2)
    l2_conv4 = model_params.get('l2_conv4', 1e-2)
    l2_dense = model_params.get('l2_dense', 1e-4)
    learning_rate = model_params.get('learning_rate', 1e-3)
    dense_size = model_params.get('dense_size', 4096)
    momentum = model_params.get('momentum', 0.5)
    decay = model_params.get('decay', 0.01)
    loss = model_params.get('loss', 'binary_crossentropy')

    input_shape = (IMAGES_DIM, IMAGES_DIM, 1)
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)

    model = keras.Sequential()
    initialize_weights_conv = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84)  # filters initialize
    initialize_weights_dense = keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=84)  # dense initialize
    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

    model.add(KL.Conv2D(n_filters_conv1, (filter_size_conv1, filter_size_conv1), activation='relu',
                        kernel_regularizer=l2(l2_conv1), kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, input_shape=input_shape))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(n_filters_conv2, (filter_size_conv2, filter_size_conv2), activation='relu',
                        kernel_regularizer=l2(l2_conv2), kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(n_filters_conv3, (filter_size_conv3, filter_size_conv3), activation='relu',
                        kernel_regularizer=l2(l2_conv3), kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(n_filters_conv4, (filter_size_conv4, filter_size_conv4), activation='relu',
                        kernel_regularizer=l2(l2_conv4), kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias))

    model.add(KL.Flatten())
    model.add(KL.Dense(dense_size, activation='sigmoid', kernel_regularizer=l2(l2_dense),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))

    hidden_first = model(first_input)
    hidden_second = model(second_input)

    L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([hidden_first, hidden_second])
    similarity = KL.Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)
    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)
    final_network.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return final_network


def build_vggface(**model_params):
    from keras_vggface.vggface import VGG16, RESNET50, SENET50

    dense_layer_size_1 = model_params.get('dense_size_1', 1024)
    dense_layer_size_2 = model_params.get('dense_size_2', 512)
    learning_rate = model_params.get('learning_rate', 1e-3)
    momentum = model_params.get('momentum', 0.5)
    decay = model_params.get('decay', 0.01)
    pre_trained_model = model_params.get('pre_trained_model', 'vgg16')
    dropout_prob = model_params.get('dropout_prob', 0.2)
    use_second_dense_layer = model_params.get('use_second_dense_layer', False)
    loss = model_params.get('loss', 'binary_crossentropy')

    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

    initialize_weights = keras.initializers.glorot_uniform(seed=84)

    input_shape = (224, 224, 3)
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)

    # remove the classifier layers and freeze the other layers
    if pre_trained_model == 'vgg16':
        vggface = VGG16()
        for i in range(6):
            vggface.layers.pop()
    elif pre_trained_model == 'resnet50':
        vggface = RESNET50()
        vggface.layers.pop()
    elif pre_trained_model == 'senet50':
        vggface = SENET50()
        vggface.layers.pop()
    else:
        raise Exception('Pretrained {} not familiar'.format(model_params['pre_trained_model']))

    for layer in vggface.layers:
        layer.trainable = False

    new_model = keras.Sequential()
    new_model.add(vggface)
    new_model.add(KL.Dense(dense_layer_size_1, activation='relu', kernel_initializer=initialize_weights,
                           bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
    new_model.add(KL.BatchNormalization())
    new_model.add(KL.Dropout(dropout_prob))
    if use_second_dense_layer:
        new_model.add(KL.Dense(dense_layer_size_2, activation='relu', kernel_initializer=initialize_weights,
                               bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        new_model.add(KL.Dropout(dropout_prob))

    first_hidden = new_model(first_input)
    second_hidden = new_model(second_input)

    L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([first_hidden, second_hidden])
    similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights,
                          bias_initializer=initialize_bias)(L1_distance)

    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    final_network.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return final_network


def euclidean_distance(vects):
    """
    calculate euclidean distance between vects[0] and vects[1]
    :param vects: vectors tuple
    :return: euclidean distance
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 2
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

