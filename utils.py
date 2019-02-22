import numpy as np
import pandas as pd

import keras
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping

def create_dataset(train_data, train_labels, test_data, test_labels):
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def myModel(nodes=64,
            hidden_layers=1,
            inputs=10000,
            outputs=1
            verbose=2,
            save_path=None):
    
    model = models.Sequential()
    model.add(layers.Dense(nodes, activation='relu', input_shape=(inputs,)))
    for layer in range(hidden_layers):
        model.add(layers.Dense(nodes, activation='relu'))
    if outputs == 1:
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.add(layers.Dense(outputs, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    val_hist = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=3,
                    batch_size=256,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                    verbose=verbose)
    
    test_hist = model.evaluate(x_test, y_test)
    model.save(save_path + '/{}/trained_model_{}_nodes_{}_layers'\
               .format(dataset, nodes, hidden_layers))
    return val_hist, test_hist

def plot_results(df):
    matrix = np.array(df)
    plt.figure(figsize=(5,5), dpi=150)
    plt.suptitle('Neural Network Accuracy Over Various Architectures', fontsize=12)
    plt.title('Best of 20 Epochs (Test Accuracy) on TensorFlow DNN (IMDB Dataset)', fontsize=8)
    im = plt.imshow(np.flip(matrix,0), interpolation='bilinear')
    plt.yticks(ticks=[0,1,2,3,4,5,6,7,8,9], labels=[1,2,4,8,16,32,64,128,256,512][::-1])
    plt.ylabel("Nodes per Hidden Layer")
    plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9], labels=[1,2,3,4,5,6,7,8,9,10])
    plt.xlabel('Hidden Layers')

    values = np.unique(matrix)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="{l:.2%}".format(l=values[i]) ) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches[::-10], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.show()