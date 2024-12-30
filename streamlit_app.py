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
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Example usage
input_sms = "Congratulations! You've won a free gift!"
transformed_sms = transform_text(input_sms)
print(transformed_sms)