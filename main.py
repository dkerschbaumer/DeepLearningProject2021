from autoencoder import Autoencoder

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.model_selection import ParameterGrid


USE_SAVED_MODEL = False


def normalizeAndReshape(data):
    data = data.astype("float32") / 255
    data = data.reshape(len(data), 28, 28, 1)
    return data

def toCategorical(data):
    return keras.utils.to_categorical(data)


def findBestHyperParameters(x_train_dist, x_train_clean, x_val_dist, x_val_clean):

    parameters = {
        'layer_type':  ['dense', 'conv'],
        'nr_layers':  [2, 3, 4],
        'nr_neurons': [32, 64, 128],
        'loss_func': ['binary_crossentropy', 'mse'],
    }
    hyperparameters = list(ParameterGrid(parameters))
    comb_list = []

    for parameter in hyperparameters:
        nr_layers = parameter['nr_layers']
        nr_neurons = parameter['nr_neurons']
        layer_type = parameter['layer_type']
        loss_func = parameter['loss_func']

        autoencoder = Autoencoder(neurons=nr_neurons, activation='relu', input_size=28)

        if layer_type == 'dense':
            autoencoder.createDenseModel(nr_layers, loss_func)
        else:
            autoencoder.createConvolutionalModel(nr_layers, loss_func)

        print(nr_layers)
        print(nr_neurons)
        print(layer_type)
        print(loss_func)

        history = autoencoder.train(x_train_dist, x_train_clean, x_val_dist, x_val_clean, batch_size=32, epochs=20)

        print(history)

        entry = {
            'nr_layers': nr_layers,
            'nr_neurons': nr_neurons,
            'layer_type': layer_type,
            'loss_func': loss_func,
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'acc': history.history['accuracy'],
            'val_acc': history.history['val_accuracy'],
        }
        comb_list.append(entry)

    best_comb = {}
    for entry in comb_list:
        if not best_comb or best_comb['val_acc'] > entry['val_acc']:
            best_comb = entry

    return best_comb


def createAndTrainModel(best_hyperparameters, x_train_dist, x_train_clean):
    autoencoder = Autoencoder(neurons=64, activation='relu', input_size=28)

    if best_hyperparameters['layer_type'] == 'dense':
        autoencoder.createDenseModel(best_hyperparameters['nr_layers'], best_hyperparameters['loss_func'])
    else:
        autoencoder.createConvolutionalModel(best_hyperparameters['nr_layers'], best_hyperparameters['loss_func'])

    autoencoder.train(x_train_dist, x_train_clean, None, None, batch_size=32, epochs=20, validation_set=False)
    return autoencoder


def printImg(img, title):
    plt.imshow(img, cmap='Greys')
    plt.title(title + str(img.shape))
    plt.show()


def plotEvaluation(autoencoder, x_test_dist, x_test_clean, y_test_clean):

    prediction = autoencoder.autoencoder.predict(x_test_dist)

    # plot side descriptions of the dataset
    plt.figure(figsize=(20, 4))
    plt.subplot(3, 11, 1)
    plt.text(0,0.5,"Original Data")
    plt.axis('off')

    plt.subplot(3, 11, 12)
    plt.text(0,0.5,"Disturbed Data")
    plt.axis('off')

    plt.subplot(3, 11, 23)
    plt.text(0,0.5,"Reconstructed Data")
    plt.axis('off')

    for i in range(0, 10, 1):
        # plot original data
        plt.subplot(3, 11, i + 2)
        plt.imshow(x_test_clean[i, :, :], cmap='Greys')
        curr_lbl = y_test_clean[i]
        plt.title("(Label: " + str(curr_lbl) + ")")

        # plot distorted data
        plt.subplot(3, 11, i + 13)
        plt.imshow(x_test_dist[i, :, :], cmap='Greys')

        # plot reconstructed data
        plt.subplot(3, 11, i + 24)
        plt.imshow(prediction[i, :, :], cmap='Greys')

    plt.show()


def main():

    # disturbed data set
    x_train_dist_full = np.load('data/distorted/X_kannada_MNIST_train_single_distorted.npy')
    x_test_dist = np.load('data/distorted/X_kannada_MNIST_test_single_distorted.npy')
    x_train_dist_full = normalizeAndReshape(x_train_dist_full)
    x_test_dist = normalizeAndReshape(x_test_dist)

    # clean data set
    x_train_clean_full = pd.read_csv('data/train.csv').iloc[:, 1:].to_numpy() # clean dataset
    y_train_clean_full = pd.read_csv('data/train.csv').iloc[:, 0].to_numpy() # label of the image
    x_train_clean_full = normalizeAndReshape(x_train_clean_full)
    y_train_clean_full = toCategorical(y_train_clean_full)

    # the test.csv does not contain a label (only the id of an image for kaggle), which means it is useless for evaluation
    # y_test_clean = pd.read_csv('data/test.csv').iloc[:, 0].to_numpy() # id of image
    x_test_clean = pd.read_csv('data/Dig-MNIST.csv').iloc[:, 1:].to_numpy() # dataset
    y_test_clean = pd.read_csv('data/Dig-MNIST.csv').iloc[:, 0].to_numpy() # target
    x_test_clean = normalizeAndReshape(x_test_clean)
    # y_test_clean = toCategorical(y_test_clean)


    # "distorted x" should be predicted as "y clean"
    # create validation set
    x_train_dist, x_val_dist, y_train_clean, y_val_clean = train_test_split(x_train_dist_full, y_train_clean_full, test_size=0.2, random_state=0)

    # normal x should also be predicted to "y clean"
    # normal data
    x_train_clean, x_val_clean, _ , _ = train_test_split(x_train_clean_full, y_train_clean_full, test_size=0.2,random_state=0)
                            # we already have this split


    autoencoder = Autoencoder(neurons=64, activation='relu', input_size=28)
    if USE_SAVED_MODEL:
       autoencoder.autoencoder = load_model('data/encoder/autoencoder.h5')

    else:

        # best_hyperparameters = findBestHyperParameters(x_train_dist, x_train_clean, x_val_dist, x_val_clean)

        best_hyperparameters = {
            'layer_type' : 'dense',
            'nr_layers' : 3,
            'loss_func' : 'binary_crossentropy'
        }
        print(best_hyperparameters)

        # best combination, but does not look good:
        # nr_layers 3
        # nr_neurons 128
        # layer_type conv
        # loss_func binary_crossentropy


        autoencoder = createAndTrainModel(best_hyperparameters, x_train_dist_full, x_train_clean_full)


    autoencoder.evaluate(x_test_dist, x_test_clean)
    plotEvaluation(autoencoder, x_test_dist, x_test_clean, y_test_clean)

if __name__ == "__main__":
    main()