import math

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping

PLOT = False


class Autoencoder:
    def __init__(self, neurons, activation, input_size):
        self.neurons = neurons
        self.activation = activation
        self.input_size = input_size
        self.full_size = input_size * input_size

        self.encoder = None
        self.autoencoder = None
        pass

    def createDenseModel(self, nr_layers, loss_func, regs=None):

        if regs['reg1'] == 'l1':
            reg1 = keras.regularizers.L1(regs['weight1'])
        elif regs['reg1'] == 'l2':
            reg1 = keras.regularizers.L2(regs['weight1'])
        elif regs['reg1'] == 'l1l2':
            reg1 = keras.regularizers.L1L2(l1=regs['weight1'], l2=regs['weight1'])
        else:
            reg1 = None

        if regs['reg2'] == 'l1':
            reg2 = keras.regularizers.L1(regs['weight2'])
        elif regs['reg2'] == 'l2':
            reg2 = keras.regularizers.L2(regs['weight2'])
        elif regs['reg2'] == 'l1l2':
            reg2 = keras.regularizers.L1L2(l1=regs['weight2'], l2=regs['weight2'])
        else:
            reg2 = None

        encoder_input = keras.Input(shape=(self.input_size, self.input_size, 1), name='Image Import')
        hidden1 = keras.layers.Flatten()(encoder_input)  # converts from 2d (30x30) to 1d (900)
        encoded = keras.layers.Dense(512, activation='relu', kernel_regularizer=reg1)(hidden1)
        encoded = keras.layers.Dense(512, activation='relu', kernel_regularizer=reg1)(encoded)
        for i in range(2, nr_layers):
            encoded = keras.layers.Dense(256, activation='relu', kernel_regularizer=reg1)(encoded)
        encoder_output = keras.layers.Dense(self.neurons, activation=self.activation, kernel_regularizer=reg1)(encoded)
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')

        decoder_input = keras.layers.Dense(self.full_size, activation='sigmoid', kernel_regularizer=reg1)(encoder_output)
        decoder_output = keras.layers.Reshape((self.input_size, self.input_size, 1))(decoder_input)

        self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='Adam', loss=loss_func, metrics=['accuracy'])

    def createConvolutionalModel(self, nr_layers, loss_func):
        encoder_input = keras.Input(shape=(self.input_size, self.input_size, 1), name='Image Import')

        encoded = keras.layers.Conv2D(self.neurons, (3, 3), activation=self.activation, padding='same')(encoder_input)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
        for i in range(1, nr_layers):
            encoded = keras.layers.Conv2D(self.neurons, (3, 3), activation=self.activation, padding='same')(encoded)
            encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
        encoded = keras.layers.Flatten()(encoded)
        encoded = keras.layers.Dense(784, activation='softmax')(encoded)

        decoded = keras.layers.Reshape((28, 28, 1))(encoded)
        decoded = keras.layers.Conv2DTranspose(self.neurons, (3, 3), activation=self.activation, padding='same')(
            decoded)
        decoded = keras.layers.BatchNormalization()(decoded)
        for i in range(1, nr_layers):
            decoded = keras.layers.Conv2DTranspose(self.neurons, (3, 3), activation=self.activation, padding='same')(
                decoded)
            decoded = keras.layers.BatchNormalization()(decoded)
        decoder_output = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

        self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='Adam', loss=loss_func, metrics=['accuracy'])

    def createModel(self, layer_type, nr_layers, loss_func):

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

        # hidden1 = keras.layers.Flatten()(encoder_input)  # converts from 2d (30x30) to 1d (900)
        # encoded = keras.layers.Dense(512, activation='relu')(hidden1)
        # encoded = keras.layers.Dense(512, activation='relu')(encoded)
        # encoded = keras.layers.Dense(256, activation='relu')(encoded)
        # encoded = keras.layers.Dense(256, activation='relu')(encoded)

        # encoder_output = keras.layers.Dense(self.neurons, activation=self.activation)(encoded)
        # self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')  # not sure if we need it really

        # DECODER
        # decoder_input = keras.layers.Dense(self.full_size, activation='sigmoid')(encoder_output)
        # decoder_output = keras.layers.Reshape((self.input_size, self.input_size, 1))(decoder_input)
        # self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
        # self.autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
        pass

    def train(self, x_train_dist, x_train_clean, x_val_dist, x_val_clean, batch_size, epochs, validation_set=True):

        if validation_set:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
            history = self.autoencoder.fit(x_train_dist, x_train_clean, epochs=epochs, batch_size=batch_size,
                                           shuffle=True,
                                           validation_data=(x_val_dist, x_val_clean), callbacks=[early_stopping])

        else:
            early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto')
            history = self.autoencoder.fit(x_train_dist, x_train_clean, epochs=epochs, batch_size=batch_size,
                                           shuffle=True,
                                           callbacks=[early_stopping])
        if PLOT:
            plt.plot(history.history['loss'], label='train')
            if validation_set:
                plt.plot(history.history['val_loss'], label='val')
            plt.title("Loss")
            plt.legend()
            plt.show()

            plt.plot(history.history['accuracy'], label='train')
            if validation_set:
                plt.plot(history.history['val_accuracy'], label='val')
            plt.title("Accuracy")
            plt.legend()
            plt.show()

        self.autoencoder.save('data/models/autoencoder.h5')

        return history

    def evaluate(self, x_test, y_test):
        results = self.autoencoder.evaluate(x_test, y_test)
        print("test loss, test acc:", results)

        pass

    # example call: autoencoder.predictExample(x_train[0])
    def predictExample(self, input_data):
        input_reshaped = input_data.reshape(-1, self.input_size, self.input_size, 1)
        res = self.autoencoder.predict([input_reshaped])
        plt.imshow(res[0], cmap='gray')
        plt.title("Example prediction " + str(res[0].shape))
        plt.show()
        return res

    def showEncoderPrediction(self, input_data):
        input_reshaped = input_data.reshape(-1, self.input_size, self.input_size, 1)
        example = self.encoder.predict([input_reshaped])
        size = math.sqrt(self.neurons)
        size = int(size)
        plt.imshow(example[0].reshape((size, size)), cmap="gray")
        plt.title("Encoder example " + str(size) + 'x' + str(size))
        plt.show()
