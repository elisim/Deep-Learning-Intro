import src.models as models
from src.lfw_dataset import LFWDataLoader, _load_image_vgg
import tensorflow as tf
import keras
from keras.regularizers import l2
from sklearn.externals import joblib
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

# for CUBLAS_STATUS_ALLOC_FAILED error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class Siamese:
    def __init__(self):
        self.model_type = None
        self.model = None
    
    def build(self, model='hani', **model_params):
        """
        :param model: model name
        :param model_params: model params
        :return: return the model with the given params
        """
        self.model_type = model
        model = getattr(models, 'build_' + model)
        if model_params:
            network = model(**model_params)
        else:
            network = model()

        self.model = network
        return network
    
    def train(self, 
              same_train_paths,       
              diff_train_paths, 
              same_val_paths, 
              diff_val_paths, 
              batch_size=32, 
              epochs=40, 
              epoch_shuffle=False, 
              earlystop_patience=10,
              verbose=2,
              use_allocated_pairs=False,
              use_worst_pairs=True,
              size_allocated_pairs=12, 
              model=None):
        
        
        if self.model_type == 'vggface':
            training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=epoch_shuffle, batch_size=batch_size, channels=3, load_image_func=_load_image_vgg, dim=(224,224), use_allocated_pairs=use_allocated_pairs, use_worst_pairs=use_worst_pairs, size_allocated_pairs=size_allocated_pairs, model=model)
            validation_generator = LFWDataLoader(same_val_paths, diff_val_paths, shuffle=epoch_shuffle, batch_size=batch_size, channels=3, load_image_func=_load_image_vgg, dim=(224,224))            
        else:
            training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=epoch_shuffle, batch_size=batch_size, use_allocated_pairs=use_allocated_pairs, use_worst_pairs=use_worst_pairs, size_allocated_pairs=size_allocated_pairs, model=model)
            validation_generator = LFWDataLoader(same_val_paths, diff_val_paths)
    
        history = self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False, verbose=verbose, epochs=epochs,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, verbose=1)])
        
        #history = self.model.fit_generator(generator=training_generator,
        #            validation_data=validation_generator,
        #            use_multiprocessing=False, verbose=verbose, epochs=epochs)        
        
        return history  

    def predict(self, same_paths, diff_paths):
        ##### TODO do flatten on gen
        generator = LFWDataLoader(same_paths, diff_paths, batch_size=len(same_paths)*2, shuffle=False)
        X,y = list(generator)[0]
        return self.model.predict([X[0], X[1]]), y
    
    def evaluate(self, train_history, same_test_paths, diff_test_paths):
        """
        :param train_history: train history
        :param same_test_paths: same test paths
        :param diff_test_paths: different test paths
        Plots accuracy and loss of train, val and test.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        fig.set_figheight(7)
        fig.set_figwidth(14)

        # plot accuracy 
        axes[0].plot(train_history.history['acc'])
        axes[0].plot(train_history.history['val_acc'])
        axes[0].set_title('model accuracy during training')
        axes[0].set_ylabel('accuracy')
        axes[0].set_xlabel('epoch')
        axes[0].legend(['training', 'validation'], loc='best')

        # plot loss
        axes[1].plot(train_history.history['loss'])
        axes[1].plot(train_history.history['val_loss'])
        axes[1].set_title('model loss during training')
        axes[1].set_ylabel('loss')
        axes[1].set_xlabel('epoch')
        axes[1].legend(['training', 'validation'], loc='best')

        print()
        test_loss, test_accuracy = self.test(same_test_paths, diff_test_paths)
        print(f'Final test loss: {test_loss:.3}')
        print(f'Final test accuracy: {test_accuracy:.3}')
    
    def test(self, same_test_paths, diff_test_paths, epoch_shuffle=False):
        if self.model_type == 'vggface':
            test_generator = LFWDataLoader(same_test_paths, diff_test_paths, shuffle=epoch_shuffle, channels=3, load_image_func=_load_image_vgg, dim=(224,224))
        else:
            test_generator = LFWDataLoader(same_test_paths, diff_test_paths, shuffle=epoch_shuffle)
        loss, accuracy = self.model.evaluate_generator(test_generator, verbose=1)
        return loss, accuracy
    
    def run_hyperas_experiment(self):
        """
        The function execute a test different parameters for a given network architecture with bayesian
        optimization process and saves the results in the disk.
        """
        trials = Trials()
        best_run, best_model = optim.minimize(model=hyperas_build_vggface,
                                              data=lfw_dataset.load_data,
                                              algo=tpe.suggest,
                                              max_evals=100,
                                              trials=trials)
                                      
        joblib.dump(best_run, 'best_run_transfer.jblib')
        joblib.dump(trials, 'transfer_all_trials_data.jblib')

    
def hyperas_build_custom_network(same_train_paths, diff_train_paths, same_val_paths, diff_val_paths, same_test_paths, diff_test_paths):
    import keras
    K = keras.backend
    KL = keras.layers
    
	# generators
    training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=True)
    validation_generator = LFWDataLoader(same_val_paths, diff_val_paths, shuffle=True)

    input_shape = (250, 250, 1)

    model = keras.Sequential()
	
    from os.path import isdir, exists
    from lfw_dataset import _extract_samples_paths, train_info_url, test_info_url
    import numpy as np
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)    
    
    model = keras.Sequential()
    initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=84)  # filters initialize
    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

    model.add(KL.Conv2D(5, (6,6), strides=(2, 2), activation='relu', input_shape=input_shape,
					 kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(14, (6, 6), strides=(2, 2), activation='relu', input_shape=input_shape,
					 kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D(60, (6, 6), strides=(2, 2), activation='relu', input_shape=input_shape,
					 kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPool2D())

    model.add(KL.Flatten())

    model.add(KL.Dense({{choice([40, 128, 256, 512])}}, activation='relu', kernel_regularizer=l2({{uniform(0, 0.1)}}),
					kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    model.add(KL.Dense({{choice([40, 128, 256, 512])}}, activation=None, kernel_regularizer=l2({{uniform(0, 0.1)}}),
					kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

	# Generate the encodings (feature vectors) for the two images
    encoded_l = model(first_input)
    encoded_r = model(second_input)

	# calculate similarity
    L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights, bias_initializer=initialize_bias)(L1_distance)
	
    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)                                                                                                    
    optimizer = keras.optimizers.SGD(lr={{uniform(0.0001, 0.1)}}, momentum={{uniform(0,1)}}, decay=0.01)      
    final_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = final_network.fit_generator(generator=training_generator,
									  validation_data=validation_generator,
									  use_multiprocessing=False, verbose=1, epochs=30)

    validation_acc = np.amax(history.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'history': history.history}



def hyperas_build_vggface(same_train_paths, diff_train_paths, same_val_paths, diff_val_paths, same_test_paths, diff_test_paths):
    import keras
    K = keras.backend
    KL = keras.layers
    from os.path import isdir, exists
    from lfw_dataset import _extract_samples_paths, train_info_url, test_info_url
    import numpy as np
    
    
    from keras_vggface.vggface import VGGFace
    
    initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=84)  # filters initialize
    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize
    
    # generators
    training_generator = LFWDataLoaderVGG(same_train_paths, diff_train_paths, shuffle=True)
    validation_generator = LFWDataLoaderVGG(same_val_paths, diff_val_paths, shuffle=True)
    
    input_shape = (224, 224, 3)
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)
        
    vggface = VGGFace(model='vgg16')
    vggface.layers.pop()
    for layer in vggface.layers:
        layer.trainable = False
        
    new_model = keras.Sequential()
    new_model.add(vggface)
    new_model.add(KL.Dense({{choice([40, 128, 256, 512])}}, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))
    new_model.add(KL.BatchNormalization())
    new_model.add(KL.Dropout({{uniform(0, 0.1)}}))
    new_model.add(KL.Dense({{choice([40, 128, 256])}}, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))
#    new_model.add(KL.BatchNormalization())
#     new_model.add(KL.Dropout({{uniform(0, 0.1)}}))

    first_hidden = new_model(first_input)
    second_hidden = new_model(second_input)
        
    L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([first_hidden, second_hidden])
    similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights, bias_initializer=initialize_bias)(L1_distance)
	
    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)                                                                                                    
    optimizer = keras.optimizers.SGD(lr={{uniform(0.0001, 0.1)}}, momentum={{uniform(0,1)}}, decay=0.01)      
    final_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    final_network.summary()
    history = final_network.fit_generator(generator=training_generator,
									  validation_data=validation_generator,
									  use_multiprocessing=False, verbose=1, epochs=15)

    validation_acc = np.amax(history.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'history': history.history}