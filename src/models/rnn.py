import os
# Set the current working directory
os.chdir(r'C:\DSClean\NLP_Disaster_Tweets')

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import src.visualization.visualize as vis
import src.data.make_dataset as mkd


# Create TensorFlow datasets from the training and validation data
train_dataset = mkd.create_tf_dataset_from_csv('C:/DSClean/NLP_Disaster_Tweets/data/interim/train.csv')
val_dataset = mkd.create_tf_dataset_from_csv('C:/DSClean/NLP_Disaster_Tweets/data/interim/val.csv')

# Define the hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000

# Initialize the text vectorization layer
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.batch(BATCH_SIZE).map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

# Define the model architecture
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

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset.batch(BATCH_SIZE), epochs=30, validation_data = val_dataset.batch(BATCH_SIZE), validation_steps=30)

# Plot the training and validation metrics
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_epoch_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_epoch_graphs(history, 'loss')
plt.ylim(0, None)

plt.show()