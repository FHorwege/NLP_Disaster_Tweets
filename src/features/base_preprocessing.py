import pandas as pd
from sklearn.model_selection import train_test_split

# text cleaning
def text_cleaning(training_dataset, test_dataset):
    training_dataset['text'] = training_dataset['text'].apply(lambda x: x.replace('\n', ' '))
    test_dataset['text'] = test_dataset['text'].apply(lambda x: x.replace('\n', ' '))
    
    return training_dataset, test_dataset
    
def remove_urls(text):
    text = re.sub(r'http\S+', '', text)
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# tokenization
# stop words removal
# lemmatization

# Split the train dataset into train and validation datasets.
def split_train_dataset(full_training_dataset):
    
    full_training_dataset['text'] = full_training_dataset['text'].apply(lambda x: x.replace('\n', ' '))
    
    train, val = train_test_split(full_train, test_size=0.2, random_state=42, shuffle=True)
    
    train.to_csv('../../data/interim/train.csv', index=False)
    val.to_csv('../../data/interim/val.csv', index=False)

