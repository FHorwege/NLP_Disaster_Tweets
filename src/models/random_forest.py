import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import optuna
from optuna.samplers import TPESampler

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DisasterTweetPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self):
        self.data = pd.read_csv(self.data_path, encoding='ISO-8859-1')

    def preprocess_text(self, content):
        content = re.sub(r'http\S+', '', content) # Remove URLs
        content = re.sub('[^a-zA-Z]', ' ', content) # Remove non-alphabetic characters
        content = content.lower() # Convert to lowercase
        words = word_tokenize(content) # Tokenize
        processed_words = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
            for word in words if word not in self.stop_words
        ]
        return ' '.join(processed_words)

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess_data(self):
        self.data['processed_text'] = self.data['text'].apply(self.preprocess_text)

    def split_data(self):
        X = self.data['processed_text'].values
        y = self.data['target'].values
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    def vectorize_data(self):
        self.X_train_vect = self.vectorizer.fit_transform(self.X_train)
        self.X_val_vect = self.vectorizer.transform(self.X_val)

    def train_model(self, params):
        self.model = RandomForestClassifier(**params, random_state=42)
        self.model.fit(self.X_train_vect, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_val_vect)
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Confusion Matrix:\n{confusion_matrix(self.y_val, y_pred)}')
        print(f'Classification Report:\n{classification_report(self.y_val, y_pred)}')
        return accuracy

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.vectorize_data()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
    }
    predictor.train_model(params)
    return cross_val_score(predictor.model, predictor.X_train_vect, predictor.y_train, cv=3, scoring='accuracy').mean()

if __name__ == "__main__":
    data_path = r'C:\DSClean\NLP_Disaster_Tweets\data\datasets\train.csv'
    predictor = DisasterTweetPredictor(data_path)
    predictor.run()

    # Use TPESampler for efficient optimization
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=20, n_jobs=-1)  # Run trials in parallel

    print('Best hyperparameters:', study.best_params)

    # Train and evaluate the model with the best hyperparameters
    predictor.train_model(study.best_params)
    final_accuracy = predictor.evaluate_model()
    print(f'Final accuracy: {final_accuracy}')
