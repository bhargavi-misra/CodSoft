import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer
with open('genre_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Movie Genre Predictor 🎬")

# Input box for the user’s plot summary
user_input = st.text_area("Enter movie plot summary:")

if st.button("Predict Genres"):
    if user_input.strip() == "":
        st.warning("Please enter a plot summary to predict.")
    else:
        # Vectorize user input
        X_input = vectorizer.transform([user_input])
        
        # Predict
        y_pred = model.predict(X_input)
        
        # Assuming y_pred is multilabel binary matrix, get genre names
        genres = ['action', 'adult', 'adventure', 'animation', 'biography', 'comedy',
                  'crime', 'documentary', 'drama', 'family', 'fantasy', 'game-show',
                  'history', 'horror', 'music', 'musical', 'mystery', 'news',
                  'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show',
                  'thriller', 'war', 'western']
        
        predicted_genres = [genres[i] for i, val in enumerate(y_pred[0]) if val == 1]
        
        if predicted_genres:
            st.success(f"Predicted genres: {', '.join(predicted_genres)}")
        else:
            st.info("No genres predicted. Try a more descriptive plot!")
