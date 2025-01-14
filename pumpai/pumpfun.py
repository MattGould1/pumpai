import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
pd.set_option('future.no_silent_downcasting', True)

import tensorflow as tf
import keras

def main():
    # train_regression() 

    raw_dataset = pd.read_csv("./data/mints/predict_mints.csv")
    dataset = raw_dataset.copy()
    dataset = dataset.drop(columns=["mint"])

    market_caps = dataset.pop("market_cap")

    dataset['is_buy'] = dataset['is_buy'].map({False: 0.0, True: 1.0 })

    predict_with_linear_model(dataset)
    print(market_caps)
    # 47.089034494
    # 43.03559379

def train_regression():
    raw_dataset = pd.read_csv("./data/mints/all_mint.csv")
    dataset = raw_dataset.copy()
    dataset = dataset.drop(columns=["mint"])

    dataset['is_buy'] = dataset['is_buy'].map({False: 0.0, True: 1.0 })

    train_dataset = dataset.sample(frac=0.8, random_state=0)

    sns.pairplot(train_dataset[['market_cap', 'is_buy', 'sol_amount', 'token_amount']], diag_kind='kde')
    plt.savefig('output.png')

    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('market_cap')
    test_labels = test_features.pop('market_cap')

    normalizer = keras.layers.Normalization(axis=-1)

    normalizer.adapt(np.array(train_features))

    linear_model = keras.Sequential([
        normalizer,
        keras.layers.Dense(units=1)
    ])

    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging.
        # verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.2)

    plot_loss(history)

    test_results = {}
    test_results['linear_model'] = linear_model.evaluate(
        test_features, test_labels, verbose=0)

    # print(test_results)

    # print(linear_model.predict(['37700868,0.0,28.543507187,1322466000000,1000000000000000,5450.6681324295205,30274268656,1063279194906016']))
    linear_model.save('./data/models/linear_regression.keras')

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.savefig('model.png')

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
 
def predict_with_linear_model(input_data):
    model = keras.models.load_model('./data/models/linear_regression.keras')
    prediction = model.predict(input_data)
    print(prediction)
    return prediction
 