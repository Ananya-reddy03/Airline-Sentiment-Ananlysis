import streamlit as st
import re
import string
import nltk
import spacy
import pickle

# Download NLP resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model not found. Run `python -m spacy download en_core_web_sm`.")
    st.stop()

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page config
st.set_page_config(page_title="Airline Sentiment Analyzer", layout="centered")
# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("aeroplane.png", width=120)
    with st.expander("👩‍💻 About Me"):
        st.markdown("""
        Hi, I'm **Ananya**, a Data Science student at **Guru Nanak Institutions**.  
        I specialize in NLP and machine learning, and I love building clean, reproducible projects.  
        This app is part of my portfolio to showcase sentiment analysis using real-world data.
        """)
# -------------------- Title --------------------
st.markdown("## ✈️ Airline Tweet Sentiment Analyzer")
st.markdown("Real-time sentiment prediction for airline-related tweets.")

# -------------------- Input --------------------
user_input = st.text_area("Enter a tweet:", height=100, max_chars=280,
                          placeholder="e.g., '@united This flight was amazing, great service!'")

# -------------------- Prediction --------------------
if st.button("Analyze"):
    if user_input:
        cleaned_input = clean_text(user_input)
        input_vectorized = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vectorized)[0]
        proba = model.predict_proba(input_vectorized)
        confidence = round(max(proba[0]) * 100, 2)

        # Prediction result
        st.markdown("### 🔍 Prediction")
        st.write(f"**Sentiment:** {prediction.capitalize()}  |  **Confidence:** {confidence}%")

        if confidence > 50:
            st.progress(confidence / 100)

        # Optional: Cleaned text and preprocessing steps
        if len(cleaned_input) > 5:
            with st.expander("🧹 Cleaned Text"):
                st.code(cleaned_input, language="text")

            with st.expander("🧠 Preprocessing Steps Applied"):
                st.markdown("""
                - Lowercased text  
                - Removed URLs, mentions, hashtags  
                - Removed punctuation and digits  
                - Trimmed extra spaces
                """)

    else:
        st.warning("Please enter a tweet to analyze.")
# -------------------- Footer --------------------
st.markdown("---")
st.caption("Made with ❤️ by Ananya | Airline Sentiment Analysis Project | Powered by Streamlit")
