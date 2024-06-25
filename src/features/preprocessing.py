import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

   
def preprocess_tweet(tweet):
    # changing all the letters into lowercase
    tweet = tweet.lower()
    # removing URLs
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    # removing hashtags
    tweet = re.sub(r'#', '', tweet)
    # removing mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # removing punctuation
    tweet = re.sub(r'[%s]' % re.escape(string.punctuation), '', tweet)  
    # splitting the line in an array of words
    tweet = nltk.tokenize.word_tokenize(tweet)
    #filtering out everything but words -> no not really -> tjere is also '•√™' wtf
    tweet = [word for word in tweet if word.isalpha()]
    # removing the stopwords
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tweet = [word for word in tweet if not word in stop_words]
    # bringing the words to the root form
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tweet = [lemmatizer.lemmatize(word) for word in tweet]
    # adding the tweet divided in words to the array
    return ' '.join(tweet)

def preprocess_df(df, text_column='text', new_column='text_processed'):
    df[new_column] = df[text_column].apply(preprocess_tweet)
    return df
