import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Set the NLTK data path
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the directory to NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download the 'punkt' tokenizer
nltk.download('punkt', download_dir=nltk_data_dir)

# Preprocessing function
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = word_tokenize(text)  # Tokenize the text using 'punkt'

    text = [i for i in text if i.isalnum()]  # Keep alphanumeric words
    text = [i for i in text if i not in stopwords.words('english')]  # Remove stopwords
    text = [ps.stem(i) for i in text]  # Apply stemming

    return " ".join(text)

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Streamlit UI setup
st.title("SMS/Email Spam Classifier")

# Text input from user
input_sms = st.text_area("Enter the message")

# When the Predict button is pressed
if st.button('Predict'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)

    # Vectorize the input text
    vector_input = tfidf.transform([transformed_sms])

    # Make the prediction
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
