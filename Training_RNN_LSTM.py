import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data_path = "data_json"

def load_data(data_path):
    print("Data loading\n")
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Loaded Data")

    return x, y


def prepare_datasets(test_size,val_size):

    #load the data
    x, y = load_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size = val_size)

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(input_shape):


    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(64, input_shape = input_shape, return_sequences = True))
    model.add(tf.keras.layers.LSTM(64))

    model.add(tf.keras.layers.Dense(64, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(10,activation = "softmax"))

    return model

if __name__ == "__main__":


    x_train, x_val, x_test, y_train, y_val, y_test = prepare_datasets(0.25, 0.2)

    print(x_train.shape[0])

    input_shape = (x_train.shape[1],x_train.shape[2])
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    """
    # train model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50)

    # plot accuracy/error for training and validation
    #plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save("model_RNN_LSTM.h5")
    print("Saved model to disk")
    """

    model = tf.keras.models.load_model("model_RNN_LSTM.h5")
    print(model.predict(x_test[100]))
