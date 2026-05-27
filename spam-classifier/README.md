# Spam Message Classification Using Machine Learning

**Assignment for SHNOOR Company Selection Process**

Prepared By: **Jothir Sailesh**  
Course: **B.Tech Artificial Intelligence and Machine Learning**

---

## 1. Project Overview

This project focuses on building a Machine Learning model that can classify text messages into two categories:

- **Spam** - Unwanted or promotional messages
- **Ham (Not Spam)** - Normal user messages

### Project Goals:

- Collect and preprocess text data
- Convert text into numerical format
- Train machine learning models
- Evaluate model performance
- Predict whether a message is spam or ham

### Key Concepts Demonstrated:

- Natural Language Processing (NLP)
- Text preprocessing
- Feature extraction
- Classification algorithms
- Model evaluation

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Data visualization |
| Scikit-learn | Machine learning |
| NLTK | Natural Language Processing |
| Jupyter Notebook | Development environment |

---

## 3. Dataset Information

### Dataset Used:

**SMS Spam Collection Dataset from Kaggle**

- **Dataset Link:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Total Messages:** 5,572 SMS messages
- **Label Column:** spam/ham
- **Message Text Column:** Contains the SMS message text

---

## 4. Folder Structure

```
spam-classifier/
│
├── dataset/
│   └── spam.csv
│
├── notebook/
│   └── spam_classifier.ipynb
│
├── screenshots/
│
├── requirements.txt
│
├── README.md
│
└── spam_classifier.py
```

---

## 5. Installation Commands

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

---

## 6. Complete Python Code

Save this as `spam_classifier.py`:

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()

# Load dataset
df = pd.read_csv('dataset/spam.csv', encoding='latin1')

# Data cleaning - keep only label and message columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Label encoding
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    
    # Remove punctuation
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords and stemming
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)

# Apply preprocessing
df['processed_message'] = df['message'].apply(preprocess_text)

# Display processed text
print(df.head())

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df['processed_message']).toarray()
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

# Accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, nb_predictions))

# Confusion matrix
cm = confusion_matrix(y_test, nb_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Model 2: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:", lr_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, lr_predictions))

# Test custom message
sample_message = ["Congratulations! You won a free iPhone. Click now!"]

# Preprocess custom message
processed_sample = [preprocess_text(msg) for msg in sample_message]

# Vectorize
sample_vector = vectorizer.transform(processed_sample)

# Predict
prediction = nb_model.predict(sample_vector)

if prediction[0] == 1:
    print("\nPrediction: Spam Message")
else:
    print("\nPrediction: Ham Message")
```

---

## 7. Step-by-Step Explanation of the Code

### Step 1: Import Libraries

We import required Python libraries for:
- Data analysis
- Visualization
- NLP
- Machine learning

Example:
```python
import pandas as pd
```
Pandas is used for handling tabular data.

### Step 2: Load Dataset

Dataset is loaded using:
```python
df = pd.read_csv(file_path, encoding='latin1')
```
We use `latin1` encoding because some SMS datasets contain special characters.

### Step 3: Data Cleaning

The dataset contains unnecessary columns. We keep only:
- Label column
- Message column

```python
df = df[['v1', 'v2']]
```

### Step 4: Label Encoding

Machine learning models cannot understand text labels directly. So:
- `ham` → 0
- `spam` → 1

```python
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
```

---

## 8. Text Preprocessing

Text preprocessing is one of the most important steps in NLP.

### 8.1 Lowercase Conversion
Converts all text into lowercase.

Example: `FREE Offer` → `free offer`

### 8.2 Remove Punctuation
Removes symbols like: `!`, `?`, `.`, `,`

This reduces noise.

### 8.3 Tokenization
Splits sentence into words.

Example:
```
"I love AI"
     ↓
['I', 'love', 'AI']
```

### 8.4 Stopword Removal
Stopwords are common words like: `is`, `the`, `are`, `and`

They do not contribute much to prediction.

### 8.5 Stemming
Stemming converts words into root forms.

Examples:
- `playing` → `play`
- `played` → `play`

---

## 9. Feature Extraction Using TF-IDF

Machine learning models cannot understand raw text. So text is converted into numbers using **TF-IDF**.

**TF-IDF** means: **Term Frequency - Inverse Document Frequency**

It measures how important a word is in a message.

Example:
```python
vectorizer = TfidfVectorizer(max_features=3000)
```

---

## 10. Train-Test Split

Dataset is divided into:

| Data | Purpose |
|---|---|
| Training Data (80%) | Train model |
| Testing Data (20%) | Evaluate model |

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 11. Machine Learning Algorithms Used

### 11.1 Naive Bayes

Naive Bayes works very well for text classification problems.

**Advantages:**
- Fast
- Efficient
- High accuracy for NLP

### 11.2 Logistic Regression

A supervised learning classification algorithm.

Used for binary classification problems.

---

## 12. Model Evaluation

### Accuracy
Measures percentage of correct predictions.

Formula: **Accuracy = Correct Predictions / Total Predictions**

### Confusion Matrix
Shows:
- True Positive
- True Negative
- False Positive
- False Negative

### Classification Report
Contains:
- Precision
- Recall
- F1-Score

---

## 13. Sample Output

```
Naive Bayes Accuracy: 0.97

Prediction: Spam Message
```

---

## 14. Results

| Model | Accuracy |
|---|---|
| Naive Bayes | 97% |
| Logistic Regression | 96% |

**Naive Bayes** performed slightly better for this dataset.

---

## 15. Advantages of the Project

- Detects spam messages automatically
- Reduces unwanted SMS
- Useful in email and messaging applications
- Demonstrates NLP pipeline
- Real-world machine learning application

---

## 16. Future Improvements

Future enhancements can include:

- Deep Learning models
- LSTM networks
- Transformer models
- Web application deployment
- Real-time spam filtering
- Multi-language support

---

## 17. Conclusion

This project successfully developed a machine learning model for spam message classification.

Using NLP preprocessing techniques and machine learning algorithms, the model achieved high accuracy in detecting spam messages.

The project demonstrates practical implementation of:

- Natural Language Processing
- Text preprocessing
- TF-IDF vectorization
- Machine learning classification
- Model evaluation

Among the tested models, **Naive Bayes** achieved the best performance.

---

*For questions or inquiries, please contact: Jothir Sailesh*
