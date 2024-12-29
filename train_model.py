
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset (replace this with your actual dataset)
X_train = ["Free money!!!", "Hello, how are you?", "Call me now", "Win a prize now!"]
y_train = [1, 0, 0, 1]  # 1 for spam, 0 for not spam

# Preprocessing function
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = word_tokenize(text)

    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Preprocess the data
X_train = [transform_text(text) for text in X_train]

# Convert text data to numerical data using TF-IDF Vectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Model trained and saved successfully!")
