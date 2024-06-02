import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def save_data(data, file_path):
    data.to_csv(file_path, index=False)
    
def clean_data(data):
    data['text'] = data['text'].apply(lambda x: x.replace('\n', ' '))
    return data