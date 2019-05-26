import tensorflow as tf
from keras.regularizers import l2
keras = tf.keras
K = keras.backend
KL = keras.layers


class Siamese():

    def __init__(self, lr=1e-4, 
                       metrics=['binary_accuracy', 'accuracy']):
        self.image_dim = 250
        self.lr = lr
        self.metrics = metrics

    def init_network(self):
        # TODO: add weight init
        # TODO: add regularization
        # TODO: add learning schedule

        input_shape = (self.image_dim, self.image_dim, 1)
        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)

        model = keras.Sequential()
        initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=84)  # filters initialize
        initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

        model.add(KL.Conv2D(5, (6, 6), strides=(2, 2), activation='relu', input_shape=input_shape,
                         kernel_initializer=initialize_weights, kernel_regularizer=l2(1e-2)))
        # model.add(KL.BatchNormalization())
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(14, (6, 6), strides=(2, 2), activation='relu', kernel_initializer=initialize_weights,
                         bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        #model.add(KL.BatchNormalization())
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(60, (6, 6), activation='relu', kernel_initializer=initialize_weights,
                         bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        # model.add(KL.BatchNormalization())
        model.add(KL.MaxPool2D())

        # model.add(KL.Conv2D(256, (4, 4), activation='tanh', kernel_initializer=initialize_weights,
        #                  bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        model.add(KL.Flatten())

        model.add(KL.Dense(40, activation='relu', kernel_regularizer=l2(1e-4),
                        kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

        model.add(KL.Dense(40, activation=None, kernel_regularizer=l2(1e-4),
                        kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(first_input)
        encoded_r = model(second_input)

        # calculate similarity
        L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        similarity = KL.Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

        final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)

        optimizer = keras.optimizers.SGD(self.lr)
        final_network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['binary_accuracy'])
        return final_network