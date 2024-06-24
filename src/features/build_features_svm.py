import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from src.features.preprocessing import preprocess_df
from numpy import array


vectorizer = TfidfVectorizer(max_features=5000)


# defo check https://github.com/Gunjitbedi/Text-Classification/blob/master/Topic%20Classification.py also encoder!!
# function to vectorize the text data using TF-IDF
# function to train and evaluate the SVM model
def train_svm(text_array, target_array):
    X_train, X_val, y_train, y_val = train_test_split(text_array, target_array, test_size=0.2, stratify=target_array, random_state=42) #added strafication
    
    with open("output_train.txt", "w") as txt_file:
        for tweet in X_train:
            txt_file.write(tweet + "\n")
            
    with open("output_validation.txt", "w") as txt_file:
        for tweet in X_val:
            txt_file.write(tweet + "\n")
    
    #check thid ggf add if statement, #min_df=2
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    
    param_grid = [
        {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}, #C till 1000 mayb too much
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['sigmoid'], 'coef0': [0.1, 0.5, 1]},
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['poly'], 'coef0': [0.1, 0.5, 1],'degree': [2, 3]}
    ]
    
    
    # initializing the SVM model
    # check if kernel linear is the best option
    # svm_model = SVC(C=1.0, kernel='linear', gamma='auto')  # linear kernel #code line above is directly linear svc. brings different score even if you have normal svc and set kernel to linear, s. notes in ds project folder
    #svm_model = LinearSVC(C=1.0, max_iter=10000)
    #svm_model = SVC(C=1, gamma=0.4).fit(X_train, y_train) #default kernel rbf
    # training the model
    #svm_model.fit(X_train, y_train)
    svm_model, grid_results = param_grid_search(X_train, y_train, param_grid)
   
    # predict on the test set
    y_pred_val = svm_model.predict(X_val)
    y_pred_train = svm_model.predict(X_train)

    # Evaluate the model
    print("Accuracy of the prediction:", accuracy_score(y_val, y_pred_val))
    print("Classification Report of prediction:\n", classification_report(y_val, y_pred_val))
    print("Accuracy of the train portion:", accuracy_score(y_train, y_pred_train))
    print("Classification Report of train portion:\n", classification_report(y_train, y_pred_train))

    return svm_model, y_val, y_pred_val, grid_results

def param_grid_search(X_train, y_train, param_grid):
    grid_search = GridSearchCV(SVC(), param_grid, verbose=3, cv=5, scoring='accuracy', refit=True)

    # Fit the GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy Score:", grid_search.best_score_)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv("/Users/oyabazer/Documents/Uni/Data Science in Practice/NLP_Disaster_Tweets/docs/grid_search_results.csv", index=False)

    return best_model, grid_results


def predict_test(df_train, df_test):
    df_train = preprocess_df(df_train)
    df_test = preprocess_df(df_test) 
   
    svm_model, y_val, y_pred_val, grid_results = train_svm(df_train['text_processed'], df_train['target'])
    print(svm_model.get_params())
    
    X_test = vectorizer.transform(df_test['text_processed'])
    predictions = svm_model.predict(X_test)
    
    return predictions, grid_results


'''
def model_svm(df):
    # combining tokenized words into a single string for each tweet
    X = df['text_processed'].apply(lambda x: ' '.join(x))
    # extracting values from target column and stroring in a variable
    y = df['target']
    
    # splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # feature extraction using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
   
    
    # building and training the SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train)
    
    # making predictions on the validation set
    y_pred = svm_model.predict(X_val_tfidf)
    
    # evaluating the model
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    print("Classification Report:\n", report)
    print("Accuracy Score:", accuracy)
    
    return svm_model'''