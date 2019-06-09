import src.models as models
from src.lfw_dataset import LFWDataLoader, _load_image_vgg, load_data
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
    
        #history = self.model.fit_generator(generator=training_generator,
        #            validation_data=validation_generator,
        #            use_multiprocessing=False, verbose=verbose, epochs=epochs,
        #            callbacks=[keras.callbacks.EarlyStopping(patience=10, verbose=1)])
        
        history = self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False, verbose=verbose, epochs=epochs)        
        
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
        best_run, best_model = optim.minimize(model=hyperas_build_hani,
                                              data=load_data,
                                              algo=tpe.suggest,
                                              max_evals=100,
                                              trials=trials)
                                      
        joblib.dump(best_run, 'best_run_transfer.jblib')
        joblib.dump(trials, 'transfer_all_trials_data.jblib')




    
def hyperas_build_hani(same_train_paths, diff_train_paths, same_val_paths, diff_val_paths, same_test_paths, diff_test_paths):
    import keras
    K = keras.backend
    KL = keras.layers
    
	# generators
    training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=True)
    validation_generator = LFWDataLoader(same_val_paths, diff_val_paths)

    input_shape = (250, 250, 1)
	
    from datetime import datetime
    from os.path import isdir, exists
    from src.lfw_dataset import _extract_samples_paths, train_info_url, test_info_url
    import numpy as np
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)        
    
    def tanh_scaled(x):
        A = 1.7159
        B = 2 / 3
        return A * K.tanh(B * x)

    model_params = {}
    
    act = model_params.get('act', tanh_scaled)
    dropout = model_params.get('dropout', 0)
    batchnorm = model_params.get('batchnorm', False)
    #loss = model_params.get('loss', contrastive_loss)
    learning_rate = model_params.get('learning_rate', 1e-3)
    first_input = KL.Input(input_shape)
    second_input = KL.Input(input_shape)

    batchnorm = True
    dropout = True    
    act='relu'   
    
    model = keras.Sequential()
    
    initialize_weights_conv = keras.initializers.glorot_uniform(seed=84)  # filters initialize
    initialize_weights_dense = keras.initializers.glorot_uniform(seed=84)  # dense initialize  
    initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

    model.add(KL.Conv2D({{choice([5, 10, 14, 30, 60])}}, (6, 6), strides=(2, 2), activation=act, input_shape=input_shape,
                        kernel_initializer=initialize_weights_conv, kernel_regularizer=l2({{uniform(0, 0.1)}})))

    #model.add(KL.BatchNormalization())
    model.add(KL.Dropout({{uniform(0, 0.5)}}))
    model.add(KL.MaxPool2D())

    model.add(KL.Conv2D({{choice([5, 10, 14, 30, 60])}}, (6, 6), strides=(2, 2), activation=act, kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))
    #model.add(KL.BatchNormalization())
    model.add(KL.Dropout({{uniform(0, 0.5)}}))
    model.add(KL.MaxPool2D())

    model.add(KL.Dropout(dropout))
    model.add(KL.Conv2D({{choice([5, 10, 14, 30, 60])}}, (6, 6), activation=act, kernel_initializer=initialize_weights_conv,
                        bias_initializer=initialize_bias, kernel_regularizer=l2({{uniform(0, 0.1)}})))

    #model.add(KL.BatchNormalization())
    model.add(KL.Dropout({{uniform(0, 0.5)}}))
    model.add(KL.MaxPool2D())

    model.add(KL.Flatten())

    model.add(KL.Dense({{choice([40, 64, 128, 256])}}, activation=act, kernel_regularizer=l2({{uniform(0, 0.1)}}),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))
    model.add(KL.Dropout({{uniform(0, 0.5)}}))
    model.add(KL.Dense({{choice([40, 64, 128, 256])}}, activation=None, kernel_regularizer=l2({{uniform(0, 0.1)}}),
                       kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))
    model.add(KL.Dropout({{uniform(0, 0.5)}}))
        
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(first_input)
    encoded_r = model(second_input)

    
    # calculate similarity
    L1_distance = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_l, encoded_r])
    similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias)(L1_distance)
    final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)                                                                         
    optimizer = keras.optimizers.SGD(lr={{uniform(0.0001, 0.1)}}, momentum={{uniform(0,1)}}, decay={{uniform(0,0.1)}})      
    final_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    start_time = datetime.now()
    history = final_network.fit_generator(generator=training_generator,
									  validation_data=validation_generator,
									  use_multiprocessing=False, verbose=2, epochs=30)
    end_time = datetime.now()
    
    validation_acc = np.amax(history.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    
    test_generator = LFWDataLoader(same_test_paths, diff_test_paths)
    test_loss, test_accuracy = final_network.evaluate_generator(test_generator)
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'history': history.history, 'training_time': end_time-start_time, 'test_loss': test_loss, 'test_accuracy': test_accuracy}
