import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="SMS Spam Detector", layout="centered")

import joblib
import re
import nltk
from nltk.corpus import stopwords

@st.cache_resource
def load_components():
    nltk.download('stopwords')
    return (
        joblib.load('model/tfidf_vectorizer.joblib'),
        joblib.load('model/spam_model.joblib')
    )

vectorizer, model = load_components()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = [word for word in text.split() 
            if word not in stopwords.words('english')]
    return ' '.join(words)

# --- UI ---
st.title("📱 SMS Spam Detector")
st.markdown("""
Paste any SMS message below to check if it's **spam** or **ham (genuine)**.
""")

# Input Section
with st.container(border=True):
    user_input = st.text_area(
        "**Enter your message here:**",
        placeholder="e.g. 'Congratulations! You won a free gift...'",
        height=150
    )
    
    predict_btn = st.button("Analyze Message", type="primary")

# Results Section
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter a message first!")
    else:
        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(user_input)
            text_vector = vectorizer.transform([cleaned_text])
            proba = model.predict_proba(text_vector)[0]
            is_spam = proba[1] > 0.5
            
        if is_spam:
            st.error(f" **SPAM ALERT** ({proba[1]*100:.1f}% confidence)")
            st.balloons()
        else:
            st.success(f" Genuine Message ({proba[0]*100:.1f}% confidence)")
            st.snow()
        
        st.progress(int(proba[1] * 100))
        st.caption(f"Spam likelihood: {proba[1]*100:.1f}%")

