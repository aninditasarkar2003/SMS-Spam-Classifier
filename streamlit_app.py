import os
import nltk

# Path to save NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Create the folder if it doesn't exist
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the directory to NLTK's search path
nltk.data.path.append(nltk_data_dir)

# Download the 'punkt' tokenizer (for breaking text into words and sentences)
nltk.download('punkt', download_dir=nltk_data_dir)

# Download the stopwords data (for filtering common words like "the", "is", etc.)
nltk.download('stopwords', download_dir=nltk_data_dir)

