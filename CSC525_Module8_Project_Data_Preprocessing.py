import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import spacy

# Download necessary NLTK data
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load dataset
df = pd.read_csv('customer_queries.csv')

# Data preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercase and remove non-alphabetic tokens
    tokens = [token.lower() for token in tokens if token.isalpha()]
    # Lemmatization
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Save preprocessed data
df.to_csv('preprocessed_queries.csv', index=False)
