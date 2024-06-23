import pandas as pd
import re
import csv
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from preprocess import preprocess_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
#  stopwords does not add influential meaning

# Load the dataset
file_path = r'C:\MariamFork\Tweet_Disaster_ShallowLearning_NLP\src\datasets\train.csv'
train_df = pd.read_csv(file_path, usecols=['keyword', 'text', 'target'])


# Fill NaN values in the 'text' column with an empty string
train_df['text'] = train_df['text'].fillna('')

# Convert text column to lowercase
train_df['text'] = train_df['text'].str.lower()

# Remove URLs containing 'http'
train_df['text'] = train_df['text'].apply(lambda text: re.sub(r'http\S+', '', text))

# Remove hashtags
train_df['text'] = train_df['text'].apply(lambda text: re.sub(r'#\w+', '', text))

# Remove mention tags
train_df['text'] = train_df['text'].apply(lambda text: re.sub(r'@\w+', '', text))

# Remove numbers
train_df['text'] = train_df['text'].apply(lambda text: re.sub(r'\d+', '', text))

# Remove punctuation
train_df['text'] = train_df['text'].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

# Remove extra whitespace
train_df['text'] = train_df['text'].apply(lambda text: re.sub(r'\s+', ' ', text).strip())

# Tokenize text
train_df['tokenized_text'] = train_df['text'].apply(lambda text: word_tokenize(text))

# Stem tokens
stemmer = PorterStemmer()
train_df['tokenized_text'] = train_df['tokenized_text'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# Remove stopwords
stop_words = set(stopwords.words('english'))
train_df['tokenized_text'] = train_df['tokenized_text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Printing a sample of the modified dataframe
print(train_df[['text', 'tokenized_text']].sample(20))


new_df = preprocess_dataframe(train_df)
print(new_df)

