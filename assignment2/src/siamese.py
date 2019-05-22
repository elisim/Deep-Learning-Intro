import numpy as np

import tensorflow as tf
keras = tf.keras
K = keras.backend
KL = keras.layers

class Siamese():

    def __init__(self, X, y, val_size=0, batch_size=32):
        self.X = X
        self.y = y
        self.val_size = val_size
        self.batch_size = batch_size

        self.image_dim = self.X[0][0].shape[0]


    def get_batch(self):
        # loop over our dataset X in mini-batches of size batchSize
        for i in np.arange(0, len(self.X[0]), self.batch_size):
            # inputs = [np.zeros((self.batch_size, self.image_dim, self.image_dim, 1)) for _ in range(2)]
            # ys = np.zeros((self.batch_size,))

            # yield a tuple of the current batched data and labels
            yield (self.X[0][i: i + self.batch_size], self.y[i: i + self.batch_size])


    def init_network(self):
        # TODO: add weight init
        # TODO: add regularization
        # TODO: add learning schedule

        input_shape = (self.image_dim, self.image_dim, 1)

        model = keras.Sequential()

        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)

        model.add(KL.Conv2D(64, (10, 10), input_shape=input_shape,
                            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.BatchNormalization())
        model.add(KL.Activation('relu'))

        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(128, (7, 7),
                            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.BatchNormalization())
        model.add(KL.Activation('relu'))
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(128, (4, 4),
                            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.BatchNormalization())
        model.add(KL.Activation('relu'))
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(256, (4, 4),
                            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.BatchNormalization())
        model.add(KL.Activation('relu'))

        model.add(KL.Flatten())

        model.add(KL.Dense(512, activation='sigmoid',
                           kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.Dropout(0.25))

        first_hidden = model(first_input)
        second_hidden = model(second_input)

        L1_distance = KL.Lambda(lambda hiddens: K.abs(hiddens[0] - hiddens[1]))([first_hidden, second_hidden])
        similarity = KL.Dense(1, activation='sigmoid',
                              kernel_regularizer=keras.regularizers.l2(5e-3))(L1_distance)

        final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)

        # change that
        optimizer = keras.optimizers.SGD(lr=0.1)
        # optimizer = keras.optimizers.Adam(0.006)
        final_network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        return final_network


    def fit(self):
        pass

