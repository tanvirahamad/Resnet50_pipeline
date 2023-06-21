
from keras import layers
from tensorflow.keras.models import Model


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


def resnt34(input_shape, input, list_layers, filter_size, required_class):

    x = layers.Reshape(input_shape)(input)
    #x = layers.ZeroPadding2D((3, 3))(x)
    x = layers.Conv2D(64, (7, 7), padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for i in range(len(list_layers)):
        if i == 0:
            for j in range(list_layers[i]):
                x = identity_block(x, filter_size)
        else:
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size)
            for j in range(list_layers[i]-1):
                x = identity_block(x, filter_size)
    x = layers.AveragePooling2D((2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(required_class, activation="softmax")(x)

    return output


input = layers.Input(shape=(224*224,))
list_layers = [3, 4, 6, 3]


r34 = resnt34(
    input_shape=(224, 224, 1),
    input=input,
    list_layers=list_layers,
    filter_size=64,
    required_class=10
)

m = Model(inputs=input, outputs=r34)
m.summary()

#m.save('w.h5')