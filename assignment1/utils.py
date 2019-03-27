def print_shapes(x_train, y_train, x_val, y_val, x_test, y_test):
    print(f"x_train.shape = {x_train.shape}")
    print(f"y_train.shape = {y_train.shape}\n")
    print(f"x_test.shape = {x_test.shape}")
    print(f"y_test.shape = {y_test.shape}\n")
    print(f"x_val.shape = {x_val.shape}")
    print(f"y_val.shape = {y_val.shape}\n")


def plot(to_plot, title='Title', xlabel='', ylabel=''):
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(to_plot)
    plt.show()
