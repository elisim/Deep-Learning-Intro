import numpy as np
import keras


def print_shapes(x_train, y_train, x_val, y_val, x_test, y_test):
    print(f"x_train.shape = {x_train.shape}")
    print(f"y_train.shape = {y_train.shape}\n")
    print(f"x_test.shape = {x_test.shape}")
    print(f"y_test.shape = {y_test.shape}\n")
    print(f"x_val.shape = {x_val.shape}")
    print(f"y_val.shape = {y_val.shape}\n")


def prepare_data(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Perform one-hot encoding to the labels, and reshaping to the data
    """
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    image_size = 784
    X_train = x_train.reshape(-1, image_size)
    X_test = x_test.reshape(-1, image_size)
    X_val = x_val.reshape(-1, image_size)
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def plot(to_plot, title='Title', xlabel='', ylabel=''):
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(to_plot)
    plt.show()
