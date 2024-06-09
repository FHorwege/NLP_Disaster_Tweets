import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    

train_df = pd.read_csv('../../data/interim/train.csv')
val_df = pd.read_csv('../../data/interim/val.csv')

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['text'].values, train_df['target'].values))
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df['text'].values, val_df['target'].values))

encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

sample_text = ('Car accident on the highway')
predictions = model.predict(np.array([sample_text]))