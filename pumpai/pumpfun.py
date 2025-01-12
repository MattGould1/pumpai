import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)
pd.set_option('future.no_silent_downcasting', True)

import tensorflow as tf
import keras

def main():
    train()

def train():
    train_data = pd.read_csv("./data/mints/mint_train.csv")
    train_data_labels = pd.read_csv("./data/mints/mint_label.csv")

    labels = train_data_labels.copy()
    features = train_data.copy()
    features.replace({ True: 1.0, False: 0.0}, inplace=True)
    grouped_data = features.groupby("mint")
 
    adj = []
    for i,g in grouped_data:
        g = g.drop(columns=["mint"])
        adj.append(g.values.tolist())
 

    # Normalize ragged tensors
    # https://github.com/tensorflow/tensorflow/issues/65399
    array = tf.ragged.constant(pylist=adj, ragged_rank=2, dtype=tf.float64)
    array = array.to_tensor()
    array = tf.linalg.normalize(array, axis = None)
    array = tf.convert_to_tensor(array[0])

    model = keras.Sequential([
        keras.layers.Dense(64, activation="softmax"), 
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1)
    ])

    labels = labels.drop(columns=["mint"])
    labels = tf.constant(labels.values)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])

    model.fit(array[:60], labels[:60], epochs=10, verbose=0)
    model.evaluate(array[60:], labels[60:], verbose=2)
    model.save('./data/models/dense64.keras')

def prediction():
    model = keras.models.load_model('./data/models/dense64.keras')
    prediction = model.predict([["JCD1EJkGq6PEmyYZwtNGYeUwpNvPcmF8VZzvxP1opump","10000000","false","166858044298","1000000000000000","11345.75785845603","43917486637","732965451086414"]])
    print(prediction)