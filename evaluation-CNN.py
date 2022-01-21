import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

## This file is only for evaluation purpose
# it uses the CNN from Assignment3 of David Kerschbaumer



def createModel(kernels, convolutional_layers, dropout) :
    layers = [
                 keras.layers.Conv2D(kernels, kernel_size=3, strides=1, activation="relu",
                                     input_shape=(28, 28, 1))] + \
             convolutional_layers * \
             [
                 keras.layers.Conv2D(kernels, kernel_size=3,
                                     activation="relu"),
                 keras.layers.MaxPool2D(pool_size=2),  # pool size 2 means 2x2
                 keras.layers.Dropout(dropout),
             ] + \
             [keras.layers.Flatten()] + \
             [keras.layers.Dense(512, activation='relu'), keras.layers.Dense(10, activation="softmax")]

    model = keras.models.Sequential(layers)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fitModel(x_train, y_train, x_val, y_val):

    # best parameters from hyperparameter search
    kernels = 128
    convolutional_layers = 2
    dropout = 0.2
    batch_size = 32
    max_epochs = 5


    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=3,
                                                      restore_best_weights=True)  # stops after "patience" number of no progress on validation set


    print("---------------------------------------------------------------------\n"
          "starting best parameters with: kernels %d || conv layers %d || dropout %.2f" % (
              kernels, convolutional_layers, dropout))

    model = createModel(kernels, convolutional_layers, dropout)
    model.summary()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=max_epochs,callbacks=[early_stopping_cb])


    print("---------------------------------------------------------------------\n"
          "FINAL RESULT WITH: kernels %d || conv layers %d || dropout %.2f" % (
              kernels, convolutional_layers, dropout))
    print("training set   loss: ", history.history['loss'][-1], "    acc: ", history.history['accuracy'][-1])
    print("---------------------------------------------------------------------")



    return model


def normalizeAndReshape(data):
    data = data.astype("float32") / 255
    data = data.reshape(len(data), 28, 28)
    return data

def toCategorical(data):
    return keras.utils.to_categorical(data)



def train():
    print("start training")
    x_train = pd.read_csv('data/train.csv').iloc[:, 1:].to_numpy() # clean basic training dataset from kaggle
    y_train = pd.read_csv('data/train.csv').iloc[:, 0].to_numpy() # label of the image

    x_val = pd.read_csv('data/Dig-MNIST.csv').iloc[:, 1:].to_numpy() # basic clean Dig-MNIST as valiation set
    y_val = pd.read_csv('data/Dig-MNIST.csv').iloc[:, 0].to_numpy() # target


    x_train = normalizeAndReshape(x_train)
    x_val = normalizeAndReshape(x_val)

    y_train = toCategorical(y_train)
    y_val = toCategorical(y_val)

    model = fitModel(x_train, y_train, x_val, y_val)
    model.save_weights('data/models/evaluation_model_weights.h5')

    print("finish training")
    return model

def evaluate(model):
    print("start evaluation")


    x_train_clean =  pd.read_csv('data/train.csv').iloc[:, 1:].to_numpy() # clean dataset
    y_train = pd.read_csv('data/train.csv').iloc[:, 0].to_numpy() # label of the image is the same also for distroted set
    x_train_clean = normalizeAndReshape(x_train_clean)
    y_train_one_hot = toCategorical(y_train)

    result  = model.evaluate(x_train_clean, y_train_one_hot)
    print("clean training set loss && acc", result)


    #------------------------------------------------------------

    x_train_dist = np.load('data/distorted/X_kannada_MNIST_train_multipl_distorted.npy')
    x_train_dist = normalizeAndReshape(x_train_dist)

    result  = model.evaluate(x_train_dist, y_train_one_hot)
    print("distorted training set loss && acc", result)

    #------------------------------------------------------------

    x_train_reconstructed = np.load('data/reconstructed/X_kannada_MNIST_x_train_reconstructed.npy')
    # x_train_reconstructed = normalizeAndReshape(x_train_reconstructed) # normalization not neccessary because data is already normalized and reshaped
    result  = model.evaluate(x_train_reconstructed, y_train_one_hot)
    print("reconstructed training set loss && acc", result)

    print("finish evaluation")

USE_SAFED_EVALUATION_MODEL = False
def main():

    model = None
    if USE_SAFED_EVALUATION_MODEL:
        kernels = 128
        convolutional_layers = 2
        dropout = 0.5
        model = createModel(kernels, convolutional_layers, dropout)
        model.summary()
        model.load_weights('data/models/evaluation_model_weights.h5')
    else:
        model = train()

    evaluate(model)


if __name__ == "__main__":
    main()
