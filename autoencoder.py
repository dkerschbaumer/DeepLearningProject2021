import math
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

PLOT = True
USE_SAVED_MODEL = False


class Autoencoder:
    def __init__(self, neurons, activation, input_size):
        self.neurons = neurons
        self.activation = activation
        self.input_size = input_size
        self.full_size = input_size * input_size

        self.encoder = None
        self.autoencoder = None
        pass

    def createModel(self):
        # ENCODER
        encoder_input = keras.Input(shape=(self.input_size, self.input_size, 1), name='Image Import')

        # encoded = keras.layers.Conv2D(32, (3, 3), activation=self.activation, padding='same')(encoder_input)
        # encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
        # encoded = keras.layers.Conv2D(32, (3, 3), activation=self.activation, padding='same')(encoded)
        # encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

        # decoded = keras.layers.Conv2D(32, (3, 3), activation=self.activation, padding='same')(encoded)
        # decoded = keras.layers.UpSampling2D((2, 2))(decoded)
        # decoded = keras.layers.Conv2D(32, (3, 3), activation=self.activation, padding='same')(decoded)
        # decoded = keras.layers.UpSampling2D((2, 2))(decoded)
        # decoder_output = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
        # self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

        hidden1 = keras.layers.Flatten()(encoder_input)  # converts from 2d (30x30) to 1d (900)
        encoded = keras.layers.Dense(512, activation='relu')(hidden1)
        encoded = keras.layers.Dense(512, activation='relu')(encoded)
        encoded = keras.layers.Dense(256, activation='relu')(encoded)
        encoded = keras.layers.Dense(256, activation='relu')(encoded)

        encoder_output = keras.layers.Dense(self.neurons, activation=self.activation)(encoded)
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')  # not sure if we need it really

        # DECODER
        decoder_input = keras.layers.Dense(self.full_size, activation='sigmoid')(encoder_output)
        decoder_output = keras.layers.Reshape((self.input_size, self.input_size, 1))(decoder_input)
        self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    def train(self, distorted_input, clean_input, validation_distorted, validation_clean, batch_size, epochs):
        history = self.autoencoder.fit(distorted_input, clean_input, epochs=epochs, batch_size=batch_size, shuffle=True,
                                       validation_data=(validation_distorted, validation_clean))
        if PLOT:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.legend()
            plt.show()

        # self.encoder.save('data/encoder/encoder.h5')
        self.autoencoder.save('data/encoder/autoencoder.h5')

        return history

    def evaluate(self):
        # todo check here if the model works well - accuracy score
        pass

    # example call: autoencoder.predictExample(x_train[0])
    def predictExample(self, input):
        input_reshaped = input.reshape(-1, self.input_size, self.input_size, 1)
        res = self.autoencoder.predict([input_reshaped])
        plt.imshow(res[0], cmap='gray')
        plt.title("Example prediction " + str(res[0].shape))
        plt.show()
        return res

    def showEncoderPrediction(self, input):
        input_reshaped = input.reshape(-1, self.input_size, self.input_size, 1)
        example = self.encoder.predict([input_reshaped])
        size = math.sqrt(self.neurons)
        size = int(size)
        plt.imshow(example[0].reshape((size, size)), cmap="gray")
        plt.title("Encoder example " + str(size) + 'x' + str(size))
        plt.show()


def preprocessing(x_train, y_train, x_test, y_test):
    # normalize data in [0,1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # todo adapt to our dataset
    # reshape for NN
    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    # change y sets to one-hot-encoding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def main():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train_dist = np.load('data/distorted/X_kannada_MNIST_train_distorted.npy')
    x_test_dist = np.load('data/distorted/X_kannada_MNIST_test_distorted.npy')
    y_train_ = pd.read_csv('data/train.csv').iloc[:, 0].to_numpy()
    y_test_ = pd.read_csv('data/test.csv').iloc[:, 0].to_numpy()

    x_train_norm = pd.read_csv('data/train.csv').iloc[:, 1:].to_numpy()
    x_train_norm = x_train_norm.reshape(-1, 28, 28)
    x_train_norm = x_train_norm.astype("float32") / 255
    x_train_norm = x_train_norm.reshape(len(x_train_norm), 28, 28, 1)

    x_train, y_train_proc, x_test, y_test = preprocessing(x_train_dist, y_train_, x_test_dist, y_test_)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_proc, test_size=0.2, random_state=0)

    x_train_norm, x_val_norm, y_train_norm, y_val_norm = train_test_split(x_train_norm, y_train_proc, test_size=0.2,
                                                                          random_state=0)

    autoencoder = Autoencoder(neurons=64, activation='relu', input_size=28)
    if USE_SAVED_MODEL:
        autoencoder.autoencoder = load_model('data/autoencoder.h5')
        autoencoder.encoder = load_model('data/encoder.h5')
    else:
        autoencoder.createModel()
        autoencoder.train(x_train, x_train_norm, x_val, x_val_norm, batch_size=32, epochs=20)

    autoencoder.predictExample(x_train[0])
    plt.imshow(x_train[0], cmap='gray')
    plt.title("distorted version  ")
    plt.show()

    plt.imshow(x_train_norm[0], cmap='gray')
    plt.title("clean version  ")
    plt.show()

    pass


if __name__ == "__main__":
    main()
