import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Global variables and constants
PROCESSED_DATA_PATH = r'C:\DSClean\NLP_Disaster_Tweets\data\interim\processed_train.csv'

def main():
    # Step 1: Load the preprocessed data
    processed_data = pd.read_csv(PROCESSED_DATA_PATH)

    # Step 2: Feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(processed_data['tokenized_text']).toarray()
    Y = processed_data['target']

    # Step 3: Split the data into training and validation sets using stratified sampling
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    # Step 4: Define the parameter grid for Grid Search
    param_grid = {
        'C': np.logspace(-4, 4, 20),  # Regularization parameter
        'max_iter': [100, 200, 300, 400, 500],  # Maximum number of iterations
        'penalty': ['l2', 'l1'],
        'solver': ['liblinear', 'saga','sag','lbfgs','newton-cg'],
    }

    # Step 5: Instantiate the Grid Search with Logistic Regression
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)



    # Step 6: Fit Grid Search to find the best model
    grid_search.fit(X_train, Y_train)

    # Step 7: Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best Parameters:", best_params)

    # Step 8: Make predictions on the validation set using the best model
    Y_pred = best_model.predict(X_val)

    # Step 9: Evaluate the best model
    accuracy = accuracy_score(Y_val, Y_pred)
    conf_matrix = confusion_matrix(Y_val, Y_pred)
    class_report = classification_report(Y_val, Y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

if __name__ == "__main__":
    main()

