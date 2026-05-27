# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
# Replace path if needed
file_path = 'dataset/spam.csv'

# Read CSV
# Some versions contain extra unnamed columns
# encoding latin1 avoids encoding issues

df = pd.read_csv(file_path, encoding='latin1')

# Keep only required columns
# Rename columns

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Display first rows
print(df.head())

# Check dataset info
print(df.info())

# Check null values
print(df.isnull().sum())

# Label encoding
# spam = 1
# ham = 0

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing
ps = PorterStemmer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenization
    words = text.split()

    # Remove stopwords and stemming
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]

    return ' '.join(words)


    print("\nPrediction: Ham Message")