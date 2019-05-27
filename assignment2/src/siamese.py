import tensorflow as tf
from keras.regularizers import l2
from lfw_dataset import LFWDataLoader, LFWDataLoaderVGG

import keras

#keras = tf.keras
K = keras.backend
KL = keras.layers

# for CUBLAS_STATUS_ALLOC_FAILED error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Siamese():
    def __init__(self, lr=1e-4, 
                       momentum=0.9, 
                       decay=0.01,
                       metrics=['accuracy'],
                       loss='binary_crossentropy',
                       batchnorm=False):
        self.image_dim = 250
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.metrics = metrics
        self.loss = loss
        self.batchnorm = batchnorm
    
    def build(self, model='hani', model_params=None):
        self.model_type = model
        model = getattr(self, 'build_' + model)
        if model_params:
            network = model(model_params)
        else:
            network = model()
        return network
    
    
    def hyper_hani(self):
        #TODO: implement
        pass
    
    
    def build_hani(self):
        input_shape = (self.image_dim, self.image_dim, 1)
        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)

        model = keras.Sequential()
        initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=84)  # filters initialize
        initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

        model.add(KL.Conv2D(5, (6, 6), strides=(2, 2), activation='relu', input_shape=input_shape,
                         kernel_initializer=initialize_weights, kernel_regularizer=l2(1e-2)))
        model = self._add_batchnorm(model)
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(14, (6, 6), strides=(2, 2), activation='relu', kernel_initializer=initialize_weights,
                         bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        
        model.add(KL.MaxPool2D())

        model.add(KL.Conv2D(60, (6, 6), activation='relu', kernel_initializer=initialize_weights,
                         bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        model = self._add_batchnorm(model)
        model.add(KL.MaxPool2D())

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
        optimizer = keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, decay=self.decay)
        final_network.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

        self.model = final_network
        return final_network
    
    def build_chopra(self):
        pass
    
    def build_vggface(self):
        from keras_vggface.vggface import VGGFace
    
        initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=84)  # filters initialize
        initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize
        
        input_shape = (224, 224, 3)
        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)
        
        vggface = VGGFace(model='resnet50')
        vggface.layers.pop()
        for layer in vggface.layers:
            layer.trainable = False
        
        new_model = keras.Sequential()
        new_model.add(vggface)
        new_model.add(KL.Dense(128, activation='sigmoid', kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
        new_model.add(KL.Dropout(0.25))
        
        first_hidden = new_model(first_input)
        second_hidden = new_model(second_input)
        
        L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([first_hidden, second_hidden])
        similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights, bias_initializer=initialize_bias)(L1_distance)

        final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)
        #optimizer = keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, decay=self.decay)
        optimizer = keras.optimizers.Adam(lr=self.lr)
        final_network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=self.metrics)
        
        self.model = final_network
        return final_network

    def _add_batchnorm(self, model):
        if self.batchnorm: 
            model.add(KL.BatchNormalization())
        return model
        
        
    def train(self, 
              same_train_paths,       
              diff_train_paths, 
              same_val_paths, 
              diff_val_paths, 
              batch_size=32, 
              epochs=40, 
              epoch_shuffle=False):
        
        if self.model_type == 'vggface':
            training_generator = LFWDataLoaderVGG(same_train_paths, diff_train_paths, shuffle=epoch_shuffle, batch_size=batch_size)
            validation_generator = LFWDataLoaderVGG(same_val_paths, diff_val_paths, shuffle=epoch_shuffle, batch_size=batch_size)            
        else:
            training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=epoch_shuffle, batch_size=batch_size)
            validation_generator = LFWDataLoader(same_val_paths, diff_val_paths, shuffle=epoch_shuffle, batch_size=batch_size)
    
        history = self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False, verbose=1, epochs=epochs)
        
        return history
        
    
    def test(self, same_test_paths, diff_test_paths, epoch_shuffle=False):
        test_generator = LFWDataLoader(same_test_paths, diff_test_paths, shuffle=epoch_shuffle)
        loss, accuracy = self.model.evaluate_generator(test_generator, verbose=1)
        return loss, accuracy
