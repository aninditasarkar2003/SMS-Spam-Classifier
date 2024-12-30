import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data (only once, as this is required for tokenization and stopwords)
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def transform_text(text):
    stop_words = set(stopwords.words('english'))  # Load stopwords once
    ps = PorterStemmer()
    text = text.lower()
    text = word_tokenize(text)

    # Remove non-alphanumeric tokens and stopwords
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stop_words]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Step 1: Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Example SMS input
input_sms = "Congratulations! You've won a free gift!"

# Step 2: Preprocess the input text
transformed_sms = transform_text(input_sms)

# Step 3: Vectorize the text using the loaded vectorizer
vector_input = tfidf.transform([transformed_sms])

# Step 4: Make predictions using the loaded model
result = model.predict(vector_input)[0]

# Step 5: Display the result
if result == 1:
    print("Spam")
else:
    print("Not Spam")
