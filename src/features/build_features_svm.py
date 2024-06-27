import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, make_scorer, precision_score, recall_score, f1_score
from src.features.preprocessing import preprocess_df
from numpy import array

vectorizer = TfidfVectorizer(max_features=5000) #check and ggf add min_df=2

def train_svm(text_array, target_array):
    X_train, X_val, y_train, y_val = train_test_split(text_array, target_array, test_size=0.2, stratify=target_array, random_state=42) #added strafication
    
    with open("output_train.txt", "w") as txt_file:
        for tweet in X_train:
            txt_file.write(tweet + "\n")
            
    with open("output_validation.txt", "w") as txt_file:
        for tweet in X_val:
            txt_file.write(tweet + "\n")
    
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    
    param_grid = [
        {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}, #C till 1000 mayb too much
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['sigmoid'], 'coef0': [0.1, 0.5, 1]},
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['poly'], 'coef0': [0.1, 0.5, 1],'degree': [2, 3]}
    ]
    

    svm_model, grid_results = param_grid_search(X_train, y_train, param_grid)
   
    # predicting on the test set
    y_pred_val = svm_model.predict(X_val)
    y_pred_train = svm_model.predict(X_train)

    # evaluating the model
    print("Accuracy of the prediction:", accuracy_score(y_val, y_pred_val))
    print("Classification Report of prediction:\n", classification_report(y_val, y_pred_val))
    print("Accuracy of the train portion:", accuracy_score(y_train, y_pred_train))
    print("Classification Report of train portion:\n", classification_report(y_train, y_pred_train))

    return svm_model, grid_results, y_val, y_pred_val

def param_grid_search(X_train, y_train, param_grid):
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_score': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, verbose=3, cv=5, scoring=scoring, refit='accuracy', return_train_score=True)

    # fitting the GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # printing the best hyperparameter combination and the best accuracy score from the grid search
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy Score:", grid_search.best_score_)
    
    # printing additional metric scores
    best_index = grid_search.best_index_
    print("Best Precision Score:", grid_search.cv_results_['mean_test_precision'][best_index])
    print("Best Recall Score:", grid_search.cv_results_['mean_test_recall'][best_index])
    print("Best F1 Score:", grid_search.cv_results_['mean_test_f1_score'][best_index])


    best_model = grid_search.best_estimator_
    grid_results = pd.DataFrame(grid_search.cv_results_)
    # writing the grid search results into a csv file
    grid_results.to_csv("/Users/oyabazer/Documents/Uni/Data Science in Practice/NLP_Disaster_Tweets/docs/grid_search_results.csv", index=False)

    return best_model, grid_results


def predict_test(df_train, df_test):
    df_train = preprocess_df(df_train)
    df_test = preprocess_df(df_test) 
   
    svm_model, grid_results, y_val, y_pred_val = train_svm(df_train['text_processed'], df_train['target'])   
    print(svm_model.get_params())
    
    X_test = vectorizer.transform(df_test['text_processed'])
    predictions = svm_model.predict(X_test)
    
    return predictions, grid_results, y_val, y_pred_val

