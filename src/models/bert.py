import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Problem: TypeError (something aboout protobuf which I didn't figure out how to deal with it)
import tensorflow as tf
from tensorflow.python.eager import monitoring as tf_monitoring
from tf_keras.src import backend 
from transformers import DistilBertTokenizer, TFDistilBertModel

# Clear any previous TensorFlow session
tf.keras.backend.clear_session()

# Graph
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

# Load datasets
train_df = pd.read_csv('C:/DSClean/NLP_Disaster_Tweets/data/interim/train.csv')
val_df = pd.read_csv('C:/DSClean/NLP_Disaster_Tweets/data/interim/val.csv')

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Tokenize the text data
max_length  = 128

def tokenize_text(texts):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

train_texts = train_df['text'].values
val_texts = val_df['text'].values

train_encodings = tokenize_text(train_texts)
val_encodings = tokenize_text(val_texts)

train_labels = train_df['target'].values
val_labels = val_df['target'].values

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(10000).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
)).batch(64)

# Load DistilBERT model
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Build the model
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
cls_token = bert_output[:, 0, :]
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(cls_token)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output, name='distilbert_model')

# Compile the model with optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    epochs=1,
    validation_data=val_dataset
)

# Plot the results
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()