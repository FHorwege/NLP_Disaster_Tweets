import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class DisasterTweetPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = DecisionTreeClassifier(random_state=42)
        self.stop_words = set(stopwords.words('english'))
        self.port_stem = PorterStemmer()
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

    def train_model(self):
        self.model.fit(self.X_train_vect, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_val_vect)
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Confusion Matrix:\n{confusion_matrix(self.y_val, y_pred)}')
        print(f'Classification Report:\n{classification_report(self.y_val, y_pred)}')

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.vectorize_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    data_path = r'C:\MariamFork\Tweet_Disaster_ShallowLearning_NLP\src\datasets\train.csv'
    predictor = DisasterTweetPredictor(data_path)
    predictor.run()
