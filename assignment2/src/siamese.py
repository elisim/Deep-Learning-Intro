import src.lfw_dataset
from src.lfw_dataset import LFWDataLoader, _load_image_vgg
import tensorflow as tf
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
import keras

from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

K = keras.backend
KL = keras.layers

# for CUBLAS_STATUS_ALLOC_FAILED error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class Siamese:
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
        self.model_type = None
        self.model = None
    
    def build(self, model='hani', **model_params):
        """
        :param model: model name
        :param model_params: model params
        :return: return the model with the given params
        """
        self.model_type = model
        model = getattr(self, 'build_' + model)
        if model_params:
            network = model(**model_params)
        else:
            network = model()
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
              use_worst_pairs=False, 
              size_worst_pairs=12, 
              model=None):
        
        if self.model_type == 'vggface':
            training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=epoch_shuffle, batch_size=batch_size, channels=3, load_image_func=_load_image_vgg, dim=(224,224), use_worst_pairs=use_worst_pairs, size_worst_pairs=size_worst_pairs, model=model)
            validation_generator = LFWDataLoader(same_val_paths, diff_val_paths, shuffle=epoch_shuffle, batch_size=batch_size, channels=3, load_image_func=_load_image_vgg, dim=(224,224))            
        else:
            training_generator = LFWDataLoader(same_train_paths, diff_train_paths, shuffle=epoch_shuffle, batch_size=batch_size, use_worst_pairs=use_worst_pairs, size_worst_pairs=size_worst_pairs, model=model)
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
        
    def build_paper_network(self, **model_params):
        """
        :return: the network the mentioned in the original paper.
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
        momentum = model_params.get('momentum',  0.5)
        decay = model_params.get('decay',  0.01)
        loss = model_params.get('loss', 'binary_crossentropy')
        
        input_shape = (self.image_dim, self.image_dim, 1)
        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)
        
        model = keras.Sequential()
        initialize_weights_conv = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84)  # filters initialize
        initialize_weights_dense = keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=84)  # dense initialize
        initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize
        
        model.add(KL.Conv2D(n_filters_conv1, (filter_size_conv1, filter_size_conv1), activation='relu', kernel_regularizer=l2(l2_conv1), kernel_initializer=initialize_weights_conv, bias_initializer=initialize_bias, input_shape=input_shape))
        model.add(KL.MaxPool2D())
        
        model.add(KL.Conv2D(n_filters_conv2, (filter_size_conv2, filter_size_conv2), activation='relu', kernel_regularizer=l2(l2_conv2), kernel_initializer=initialize_weights_conv, bias_initializer=initialize_bias))
        model.add(KL.MaxPool2D())
        
        model.add(KL.Conv2D(n_filters_conv3, (filter_size_conv3, filter_size_conv3), activation='relu', kernel_regularizer=l2(l2_conv3), kernel_initializer=initialize_weights_conv, bias_initializer=initialize_bias))
        model.add(KL.MaxPool2D())
        
        model.add(KL.Conv2D(n_filters_conv4, (filter_size_conv4,filter_size_conv4), activation='relu', kernel_regularizer=l2(l2_conv4), kernel_initializer=initialize_weights_conv, bias_initializer=initialize_bias))
        
        model.add(KL.Flatten())
        model.add(KL.Dense(dense_size, activation='sigmoid', kernel_regularizer=l2(l2_dense), kernel_initializer=initialize_weights_dense, bias_initializer=initialize_bias))
        
        hidden_first = model(first_input)
        hidden_second = model(second_input)
        
        L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([hidden_first, hidden_second])
        similarity = KL.Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
        
        final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)
        final_network.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        self.model = final_network
        return final_network
    
    
    def build_hani(self, **model_params):
        """
        Return a network defining the siamese network in:
        --------------------------------------------------------
        Khalil-Hani, M., & Sung, L. S. (2014). A convolutional neural
        network approach for face verification. High Performance Computing
        & Simulation (HPCS), 2014 International Conference on, (3), 707–714.
        doi:10.1109/HPCSim.2014.6903759
        """

    
        def tanh_scaled(x):
            A = 1.7159
            B = 2/3
            return A*K.tanh(B*x)
        
        act = model_params.get('act', tanh_scaled)
        dropout = model_params.get('dropout', 0)
        batchnorm = model_params.get('batchnorm', False)
        loss = model_params.get('loss', contrastive_loss)
        learning_rate = model_params.get('learning_rate', 1e-3)        
		
        input_shape = (self.image_dim, self.image_dim, 1)
        first_input = KL.Input(input_shape)
        second_input = KL.Input(input_shape)
        
        model = keras.Sequential()
        initialize_weights_conv = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=84)  # filters initialize
        initialize_weights_dense = keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=84)  # dense initialize
        initialize_bias = keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=84)  # bias initialize

        
        initialize_weights_conv = keras.initializers.glorot_uniform(seed=84)
        initialize_weights_dense = keras.initializers.glorot_uniform(seed=84)
        
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
        L2_distance = KL.Lambda(euclidean_distance)([encoded_l, encoded_r])
        #similarity = KL.Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
        
        final_network = keras.Model(inputs=[first_input, second_input], outputs=L2_distance)
        #optimizer = keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, decay=self.decay)
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        final_network.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        self.model = final_network
        return final_network
    
    def build_vggface(self, **model_params):
        from keras_vggface.vggface import VGG16, RESNET50, SENET50
    
        dense_layer_size_1 = model_params.get('dense_size_1', 1024)
        dense_layer_size_2 = model_params.get('dense_size_2', 512)
        learning_rate = model_params.get('learning_rate', 1e-3)
        momentum = model_params.get('momentum', 0.5)
        decay = model_params.get('decay', 0.01)
        pre_trained_model = model_params.get('pre_trained_model', 'vgg16')
        dropout_prob = model_params.get('dropout_prob', 0.2)
        use_second_dense_layer = model_params.get('use_second_dense_layer', False)
    
        #initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=84)  # filters initialize
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
        new_model.add(KL.Dense(dense_layer_size_1, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
        new_model.add(KL.BatchNormalization())
        new_model.add(KL.Dropout(dropout_prob))
        if use_second_dense_layer:
            new_model.add(KL.Dense(dense_layer_size_2, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(1e-2)))
            new_model.add(KL.Dropout(dropout_prob))
        
        first_hidden = new_model(first_input)
        second_hidden = new_model(second_input)
        
        L1_layer = KL.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([first_hidden, second_hidden])
        similarity = KL.Dense(1, activation='sigmoid', kernel_initializer=initialize_weights, bias_initializer=initialize_bias)(L1_distance)
        
        final_network = keras.Model(inputs=[first_input, second_input], outputs=similarity)
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        #optimizer = keras.optimizers.SGD(lr=learning_rate,decay=decay, momentum=momentum)
        final_network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=self.metrics)
        
        self.model = final_network
        return final_network


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 2
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


    
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