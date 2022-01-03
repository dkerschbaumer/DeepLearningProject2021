import math

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

PLOT = True
USE_SAVED_MODEL = True

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
        # todo not sure if Conv2d layers or Dense layers - both are used in practice it seems

        # ENCODER
        encoder_input = keras.Input(shape=(self.input_size, self.input_size, 1), name='Image Import')
        # todo maybe add here Conv2d
        hidden1 = keras.layers.Flatten()(encoder_input) # converts from 2d (30x30) to 1d (900)
        encoder_output = keras.layers.Dense(self.neurons, activation=self.activation)(hidden1)
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder') # not sure if we need it really

        # DECODER
        decoder_input = keras.layers.Dense(self.full_size , activation='relu')(encoder_output)
        decoder_output = keras.layers.Reshape((self.input_size, self.input_size, 1))(decoder_input)
        self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    def train(self, input_train, validation, batch_size, epochs):
        history = self.autoencoder.fit(input_train,input_train,epochs=epochs,batch_size=batch_size,shuffle=True,
                        validation_data=(validation,validation))
        if PLOT:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.legend()
            plt.show()

        self.encoder.save('data/encoder.h5')
        self.autoencoder.save('data/autoencoder.h5')

        return history

    def evaluate(self):
        # todo check here if the model works well - accuracy score
        pass

    # example call: autoencoder.predictExample(x_train[0])
    def predictExample(self, input):
        input_reshaped = input.reshape(-1,self.input_size,self.input_size,1)
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
    # x_train = x_train.reshape(len(x_train), 28, 28, 1)
    # x_test = x_test.reshape(len(x_test), 28, 28, 1)

    # change y sets to one-hot-encoding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def main():

    # todo change here to our dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train, x_test, y_test = preprocessing(x_train, y_train, x_test, y_test)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    autoencoder = Autoencoder(neurons=64, activation='relu', input_size=28)
    if USE_SAVED_MODEL:
        autoencoder.autoencoder = load_model('data/autoencoder.h5')
        autoencoder.encoder = load_model('data/encoder.h5')
    else:
        autoencoder.createModel()
        autoencoder.train(x_train, x_val, batch_size=32, epochs=3)

    autoencoder.showEncoderPrediction(x_train[0])
    autoencoder.predictExample(x_train[0])



    pass


if __name__ == "__main__":
    main()