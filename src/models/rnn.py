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
encoder.adapt(train_dataset.batch(BATCH_SIZE).map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])

history = model.fit(train_dataset.batch(BATCH_SIZE), epochs=10, validation_data = val_dataset.batch(BATCH_SIZE), validation_steps=30)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)