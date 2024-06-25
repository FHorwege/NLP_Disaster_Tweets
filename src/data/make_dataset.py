import pandas as pd
import tensorflow as tf

def create_tf_dataset_from_csv(file_path):
    # Read the data from the CSV file
    df = pd.read_csv(file_path)
    # Create a TensorFlow dataset from the DataFrame
    dataset = tf.data.Dataset.from_tensor_slices((df['text'].values, df['target'].values))
    return dataset