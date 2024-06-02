from src.features.preprocess import standardization
import pandas as pd
from tensorflow.keras import layers

max_features = 20000
sequence_length = 500

data = pd.read_csv('../../data/interim/train.csv')

vectorize_layer = layers.TextVectorization(
    standardize= standardization(data),
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

