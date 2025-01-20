import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load the IMDb dataset
# You can replace this with the actual path or use a built-in dataset
from sklearn.datasets import fetch_openml

print("Fetching IMDb dataset...")
data = fetch_openml("imdb", version=1, as_frame=True)
data = pd.read_csv('imdb_top_1000.csv')



# Prepare the data
reviews = data.data['review']
labels = data.target.replace({"pos": 1, "neg": 0})  # Convert labels to 1 (positive) and 0 (negative)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes model
print("Training the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer
print("Saving the model and vectorizer...")
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved as 'sentiment_model.pkl' and 'vectorizer.pkl'.")
