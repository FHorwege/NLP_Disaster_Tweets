import os
<<<<<<< HEAD
# Set the current working directory
os.chdir(r'C:\DSClean\NLP_Disaster_Tweets')

=======
import sys
# Set the current working directory
sys.path.insert(0,"C:/DSClean/NLP_Disaster_Tweets/src/visualization")
sys.path.insert(0,"C:/DSClean/NLP_Disaster_Tweets/src/data")


import visualize as vis
>>>>>>> b1f03e46649e15c5e23d7d8188b7095a5395a497
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
<<<<<<< HEAD
import src.visualization.visualize as vis
from src.visualization.visualize import plot_epoch_graphs


import src.data.make_dataset as mkd
=======


import make_dataset as mkd
>>>>>>> b1f03e46649e15c5e23d7d8188b7095a5395a497

# Create TensorFlow datasets from the training and validation data
train_dataset = mkd.create_tf_dataset_from_csv('C:/DSClean/NLP_Disaster_Tweets/data/interim/train.csv')
val_dataset = mkd.create_tf_dataset_from_csv('C:/DSClean/NLP_Disaster_Tweets/data/interim/val.csv')


# Parameters
vocab_size = 2000
embedding_dim = 128
lstm_units = 32
dropout_rate = 0.5
batch_size = 32
learning_rate = 1e-4
num_filters = 8
kernel_size = 3




# Initialize the text vectorization layer
encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
encoder.adapt(train_dataset.batch(batch_size).map(lambda text, label: text))

# Define the model
model = Sequential([
    encoder,
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    Dense(64, activation='relu'),
    Dropout(dropout_rate),
    LSTM(lstm_units),
<<<<<<< HEAD
=======
    Dropout(dropout_rate),
>>>>>>> b1f03e46649e15c5e23d7d8188b7095a5395a497
    Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# Train the model
history = model.fit(train_dataset.batch(batch_size),
                    epochs=30,
                    validation_data=val_dataset.batch(batch_size),
                    callbacks=[early_stopping])


<<<<<<< HEAD
# get val_loss and val_accuracy
val_loss, val_accuracy = model.evaluate(val_dataset.batch(batch_size))

plot_epoch_graphs(history)
=======
vis.plot_epoch_graphs(history)
print(dir(vis))
>>>>>>> b1f03e46649e15c5e23d7d8188b7095a5395a497
