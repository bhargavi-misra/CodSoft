import streamlit as st
import pickle
import re

with open('SmsSpamDetection/best_spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

st.markdown(
    """
    <style>
    .title {
        color: #ff4b4b;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #ffa500;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 18px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
    .result {
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">SMS Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect spam SMS instantly!</div>', unsafe_allow_html=True)

sms_input = st.text_area("Type your SMS here:", height=150)

if st.button("Check Spam"):
    if sms_input.strip() == "":
        st.warning("Type something first")
    else:
        cleaned_input = clean_text(sms_input)
        vect_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vect_input)[0]
        if prediction == 1:
            st.markdown('<p class="result" style="color:#ff4b4b;">Spam  - Watch out!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result" style="color:#28a745;">Ham  - All clear!</p>', unsafe_allow_html=True)
