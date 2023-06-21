
from keras import layers
from tensorflow.keras.models import Model

INPUT = layers.Input(shape=(224*224,))
INPUT_SHAPE = (224, 224, 1)
LIST_LAYERS = [3, 4, 6, 3]
REQUIRED_CLASS = 10


def identity_block(x, filter_size):
    skip_connection = x
    # x = layers.ZeroPadding2D((1, 1))(x)

    # layer 1
    x = layers.Conv2D(filter_size, (3, 3), padding="same", strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # layer 2
    x = layers.Conv2D(filter_size, (3, 3), padding="same", strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)

    # add residue
    x = layers.Add()([x, skip_connection])
    x = layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter_size):
    skip_connection = x
    # x = layers.ZeroPadding2D((1, 1))(x)

    # layer 1
    x = layers.Conv2D(filter_size, (3, 3), padding="same", strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # layer 2
    x = layers.Conv2D(filter_size, (3, 3), padding="same", strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    # convolution residue
    skip_connection = layers.Conv2D(filter_size, (1, 1), padding='valid', strides=(2, 2))(skip_connection)

    # add residue
    x = layers.Add()([x, skip_connection])
    x = layers.Activation('relu')(x)
    return x


def resnt34():
    filter_size=64
    x = layers.Reshape(INPUT_SHAPE)(INPUT)
    #x = layers.ZeroPadding2D((3, 3))(x)
    x = layers.Conv2D(64, (7, 7), padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for i in range(len(LIST_LAYERS)):
        if i == 0:
            for j in range(LIST_LAYERS[i]):
                x = identity_block(x, filter_size)
        else:
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size)
            for j in range(LIST_LAYERS[i]-1):
                x = identity_block(x, filter_size)
    x = layers.AveragePooling2D((2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(REQUIRED_CLASS, activation="softmax")(x)

    model = Model(INPUT, output)
    model.summary()
    return model

if __name__ == "__main__":
    resnet34_model = resnt34()
    #resnet34_model.save('resnet34_model.h5')