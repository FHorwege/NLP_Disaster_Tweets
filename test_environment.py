import sys
import pandas as pd
import os
from src.features.build_features_svm import predict_test, param_grid_search
from src.features.preprocessing import preprocess_df
from src.visualization.visualize import keyword_dstr

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    # main()
    """sample_text = "@user Check out this link: https://example.com #NLP #AI I love learning about Natural Language Processing!!!"
    cleaned_text = text_cleaning(sample_text, correct_spelling=True)
    print("Cleaned Text:", cleaned_text)"""
    # specifying the file paths
    df_train = pd.read_csv('/Users/oyabazer/Documents/Uni/Data Science in Practice/NLP_Disaster_Tweets/docs/train.csv')
    df_test = pd.read_csv('/Users/oyabazer/Documents/Uni/Data Science in Practice/NLP_Disaster_Tweets/docs/test.csv')
    path_to_predictions_csv = '/Users/oyabazer/Documents/Uni/Data Science in Practice/NLP_Disaster_Tweets/docs/predictions.csv' 
    path_to_grid_results_csv = '/Users/oyabazer/Documents/Uni/Data Science in Practice/NLP_Disaster_Tweets/docs/grid_search_results.csv'
    
    # getting predictions and grid results
    predictions, grid_results = predict_test(df_train, df_test)
    
    df_test['predictions'] = predictions   #predict_test(df_train, df_test) modified for being able to get the grid results
    df_test.to_csv(path_to_predictions_csv, index=False)
    #keyword_dstr(df_train)
    grid_results.to_csv("grid_search_results.csv", index=False)
    
