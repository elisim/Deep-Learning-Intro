import numpy as np

import tensorflow as tf
keras = tf.keras
K = keras.backend
KL = keras.layers

class Siamese():

    def __init__(self):
        self.image_dim = 250

    def init_network(self):
        # TODO: add weight init
        # TODO: add regularization
        # TODO: add learning schedule

        input_shape = (self.image_dim, self.image_dim, 1)

        model = keras.Sequential()

        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)

        model.add(KL.Conv2D(64, (5, 5), input_shape=input_shape, padding=2))
                            #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            #bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            #kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.Activation('relu'))
        model.add(KL.BatchNormalization())
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(128, (5, 5), padding=2))
                            #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            #bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            #kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.Activation('relu'))
        model.add(KL.BatchNormalization())
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(256, (3, 3), padding=2))
                            #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            #bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            #kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.Activation('relu'))
        model.add(KL.BatchNormalization())
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(512, (3, 3)))
                            #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84),
                            #bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84),
                            #kernel_regularizer=keras.regularizers.l2(5e-3)))
        model.add(KL.BatchNormalization())
        model.add(KL.Activation('relu'))

        model.add(KL.Flatten())

        model.add(KL.Dense(512, activation='sigmoid'))
                           #kernel_regularizer=keras.regularizers.l2(5e-3)))
        #model.add(KL.Dropout(0.25))

        first_hidden = model(first_input)
        second_hidden = model(second_input)

        L1_distance = KL.Lambda(lambda hiddens: K.abs(hiddens[0] - hiddens[1]))([first_hidden, second_hidden])
        similarity = KL.Dense(1, activation='sigmoid')(L1_distance)
                              #kernel_regularizer=keras.regularizers.l2(5e-3))(L1_distance)

        final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)

        # change that
        #optimizer = keras.optimizers.SGD(lr=0.1)
        # optimizer = keras.optimizers.Adam(0.006)
        optimizer = keras.optimizers.RMSprop(lr=1e-6, decay=0.99)
        final_network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        return final_network


    def fit(self):
        pass

