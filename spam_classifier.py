import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding='latin1')

# Keep only needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset Loaded Successfully!")
print(df.head())

# Features and labels
X = df['message']
y = df['label']

# Convert text to numbers
cv = CountVectorizer()

X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Test custom message
sample = ["Congratulations! You won a free iPhone"]

sample_data = cv.transform(sample)

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("\nPrediction: Spam Message")
else:
    print("\nPrediction: Ham Message")