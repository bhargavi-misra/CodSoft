import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Get current directory
BASE_DIR = os.path.dirname(__file__)

# Paths to model and vectorizer
model_path = os.path.join(BASE_DIR, 'genre_classifier.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Movie Genre Predictor 🎬")

# User input
user_input = st.text_area("Enter movie plot summary:")

if st.button("Predict Genres"):
    if user_input.strip() == "":
        st.warning("Please enter a plot summary to predict.")
    else:
        # Vectorize input
        X_input = vectorizer.transform([user_input])

        # Predict
        y_pred = model.predict(X_input)

        # Genre list (must match training)
        genres = ['action', 'adult', 'adventure', 'animation', 'biography', 'comedy',
                  'crime', 'documentary', 'drama', 'family', 'fantasy', 'game-show',
                  'history', 'horror', 'music', 'musical', 'mystery', 'news',
                  'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show',
                  'thriller', 'war', 'western']

        # Extract predicted genres
        predicted_genres = [genres[i] for i, val in enumerate(y_pred[0]) if val == 1]

        if predicted_genres:
            st.success(f"🎯 Predicted genres: {', '.join(predicted_genres)}")
        else:
            st.info("🤔 No genres predicted. Try a longer or more detailed plot.")
