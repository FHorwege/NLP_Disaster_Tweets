import pandas as pd
import re
import csv
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle


# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
#  stopwords does not add influential meaning

print(stopwords.words('english'))

#Data Processing
# Load the dataset
file_path = r'C:\MariamFork\Tweet_Disaster_ShallowLearning_NLP\src\datasets\train.csv'
train_data = pd.read_csv(file_path, encoding='ISO-8859-1')

#givin info about colums and rows (7613, 5) basically 7614 row  starting indext 0 and 5 columns
print(train_data.shape)

print(train_data)

#checking the distribution of target column 0: 4342 1: 3271, not  equal, should we do upsampling/downsampling to make it equal?,
# because ML model wont work fairly
target_counts = train_data['target'].value_counts()
print(target_counts)


#stemming

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    content = re.sub(r'http\S+', '', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply the stemming function to the 'text' column and create a new column 'stemmed_content'
if 'text' in train_data.columns:
    train_data['stemmed_content'] = train_data['text'].apply(stemming)

    # Print the DataFrame to see the results
    print("Data after stemming:")
    print(train_data.head())
else:
    print("The 'text' column is not found in the DataFrame.")


#seperating the data and label

X=train_data['stemmed_content'].values
Y=train_data['target'].values

print("Features (X):", X)
print("Labels (Y):", Y)

# Splitting the data into training data and validation data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

# Print shapes of the splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# Print a few examples to verify
print("Training data samples:")
print(X_train[:5])
print(Y_train[:5])


#converting the textual data to numerical data
#tells us importance of each word. meaning how many times is each word repeated in overall document. getting importance right.
# once we do that machin learning transfer this either in 1 or 0. ML model will try to understand what model is important
# (0, 12778)	0.3102991438746976
# 0 means
vectorizer=(TfidfVectorizer())

X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

print(X_train)
print(X_test)


#training the ML model

#Logistic regression> classify model, positive tweet and negative tweet

model=LogisticRegression(max_iter=10000)  #max iteration
model.fit(X_train, Y_train)

#accuracy score on the training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data :', training_data_accuracy)

#accuracy score on the test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the test data :', test_data_accuracy)

#Model accuracy= 80% not bad

# Detailed evaluation on the test data
print("Confusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))
print("Classification Report:")
print(classification_report(Y_test, X_test_prediction))

# Save the model and vectorizer for future use
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

