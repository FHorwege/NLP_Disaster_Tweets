import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load the dataset
file_path = r'C:\MariamFork\Tweet_Disaster_ShallowLearning_NLP\src\datasets\train.csv'
train_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Initialize the Porter Stemmer and WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to get the part of speech tag for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(content):
    content = re.sub(r'http\S+', '', content)
    content = re.sub(r'@\w+', '', content)
    content = re.sub(r'\d+', '', content)
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    content = content.lower()
    words = word_tokenize(content)
    processed_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in words if word not in stopwords.words('english')
    ]
    return ' '.join(processed_words)

# Apply the preprocessing function to the 'text' column
train_data['processed_text'] = train_data['text'].apply(preprocess_text)

# Separating the data and labels
X = train_data['processed_text'].values
Y = train_data['target'].values

# Splitting the data into training data and validation data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)

# Training the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = knn_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data :', training_data_accuracy)

# Accuracy score on the validation data
X_val_prediction = knn_model.predict(X_val)
test_data_accuracy = accuracy_score(Y_val, X_val_prediction)
print('Accuracy score on the validation data :', test_data_accuracy)

# Detailed evaluation on the validation data
print("Confusion Matrix:")
print(confusion_matrix(Y_val, X_val_prediction))
print("Classification Report:")
print(classification_report(Y_val, X_val_prediction))

# Store the KNN accuracy for later comparison
knn_accuracy = test_data_accuracy

# Accuracy scores from different models (replace these with your actual scores)
model_names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Naive Bayes', 'KNN']
accuracy_scores = [0.75, 0.82, 0.80, 0.78, knn_accuracy]  # Example accuracy scores with KNN

# Create a DataFrame to hold the results
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_scores
})

# Plot the grouped bar plot
plt.figure(figsize=(10, 6))
bar_width = 0.4

# Set position of bar on X axis
r1 = np.arange(len(results_df))

# Create bars
plt.bar(r1, results_df['Accuracy'], color='blue', width=bar_width, edgecolor='grey', label='Accuracy')

# Add labels
plt.xlabel('Model', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Model Comparison', fontweight='bold')
plt.xticks(r1, results_df['Model'])

# Add data labels on top of the bars
for i in range(len(results_df)):
    plt.text(i, results_df['Accuracy'][i] + 0.01, f"{results_df['Accuracy'][i]:.2f}", ha='center')

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11]  # Example values of k to try
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to find the best parameters
grid_search.fit(X_train, Y_train)

# Print the best parameters found by GridSearchCV
print("Best parameters:", grid_search.best_params_)

# Use the best estimator found by GridSearchCV
best_knn_model = grid_search.best_estimator_

# Evaluate the best model on validation data
Y_val_predicted = best_knn_model.predict(X_val)
accuracy = accuracy_score(Y_val, Y_val_predicted)
print("Accuracy on validation data with best model:", accuracy)