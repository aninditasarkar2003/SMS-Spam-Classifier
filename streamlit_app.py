import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Add the custom directory to NLTK's data path
nltk.data.path.append(r'C:\Users\ANINDITA SARKAR\AppData\Roaming\nltk_data')

# Test the download by tokenizing a sample sentence (this is optional, just to test)
sample_text = "Hello, this is a test sentence."
tokens = nltk.word_tokenize(sample_text)
print(tokens)  # You can see the tokenized words in the console

# Initialize the stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    # Predict the result
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
