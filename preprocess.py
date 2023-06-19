import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

def visulize(n, image):
    plt.figure(figsize=(32, 32))
    for i in range(n):
        plt.subplot(2, int(n / 2), i + 1)
        # ax = plt.subplot(2, int(n / 2), i + 1)
        plt.title(f"image {i}")
        # ax.get_xaxis().set_visible(False)
        plt.xlabel(f"X lable {i}")
        plt.ylabel(f"Y lable {i}")
        plt.imshow(image[int(i)].reshape(32, 32, 3))
    plt.show()


def convert_target_class(y_train, y_test):
    # convert calss vector to binary class matrices(like one hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return y_train, y_test


def normalize_split_data(x, y):
    x = x.reshape((len(x), np.prod(x.shape[1:])))
    y = y.reshape((len(y), np.prod(y.shape[1:])))

    return x, y


def data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_train.astype('float32') / 255

    y_train, y_test = convert_target_class(y_train, y_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42
    )

    x_train, y_train = normalize_split_data(x_train, y_train)
    x_test, y_test = normalize_split_data(x_test, y_test)
    x_val, y_val = normalize_split_data(x_val, y_val)

    return x_train, y_train, x_test, y_test, x_val, y_val


if _name_ == "__main__":
    x_train, y_train, x_test, y_test, x_val, y_val = data()
    visulize(10, x_train[:10])
    visulize(10, x_train[:10])