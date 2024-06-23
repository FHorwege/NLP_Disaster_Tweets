import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from preprocess import preprocess_dataframe  # Assuming preprocess_dataframe is a function to preprocess the DataFrame
from sklearn.metrics import accuracy_score, classification_report


# Load the DataFrame
from prep import train_df



# Preprocess the DataFrame (if needed)
train_df = preprocess_dataframe(train_df)

# Separate the data and labels
X = train_df['text'].values  # Assuming 'text' is the column containing tokenized text
Y = train_df['target'].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=0)

# Convert text data to numerical representation (TF-IDF vectors)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes model
clf = GaussianNB()
clf.fit(X_train_tfidf.toarray(), Y_train)  # Convert to array if sparse matrix is returned

# Predictions
y_pred = clf.predict(X_test_tfidf.toarray())

# Evaluate the model
accuracy = clf.score(X_test_tfidf.toarray(), Y_test)
print("Accuracy:", accuracy)

# Evaluate the model
accuracy = clf.score(X_test_tfidf.toarray(), Y_test)
print("Accuracy:", accuracy)

# Compute classification report
report = classification_report(Y_test, y_pred)
print("Classification Report:\n", report)


# Save the model and vectorizer for future use
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pass