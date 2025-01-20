import streamlit as st
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("sentiment_model.pkl")  # Replace with your model path
    vectorizer = joblib.load("vectorizer.pkl")  # Replace with your vectorizer path
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Streamlit app
st.title("IMDb Review Sentiment Predictor")

st.markdown("This app predicts the sentiment (positive/negative) of a movie review.")

# Input box for user to enter a review
user_review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_review.strip():
        # Preprocess and predict
        review_vectorized = vectorizer.transform([user_review])
        sentiment = model.predict(review_vectorized)[0]
        sentiment_label = "Positive" if sentiment == 1 else "Negative"

        # Display the prediction
        st.subheader(f"Predicted Sentiment: {sentiment_label}")
    else:
        st.warning("Please enter a review before clicking the button.")

st.markdown("---")
st.markdown("### About this App")
st.write("The model is trained on the IMDb dataset using a machine learning classifier. It uses text vectorization techniques to analyze the input review and predict whether the sentiment is positive or negative.")
