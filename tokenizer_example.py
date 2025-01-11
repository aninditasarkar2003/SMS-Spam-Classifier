import nltk

# Download 'punkt' resource if not already available
nltk.download('punkt')

def transform_text(text):
    tokens = nltk.word_tokenize(text)
    # Perform other transformations
    return tokens

# Example usage
input_sms = "Hello, how are you?"
transformed_sms = transform_text(input_sms)
print(transformed_sms)
