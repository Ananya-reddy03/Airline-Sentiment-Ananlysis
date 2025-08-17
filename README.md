✈️ Airline Tweet Sentiment Analyzer

A Streamlit-based web app that predicts the sentiment of airline-related tweets using a trained machine learning model. Built by **Ananya**, a Data Science student at **Guru Nanak Institutions**, this project showcases real-world NLP techniques with a clean, professional UI.

📌 Project Overview

This app analyzes tweets related to airlines and classifies them into one of three sentiment categories:
- Positive
- Negative
- Neutral
It uses a trained Logistic Regression model with TF-IDF vectorization and includes preprocessing steps to clean noisy text data.

Architecture

User Tweet → Text Cleaning → TF-IDF Vectorizer → Logistic Regression Model → Sentiment Output

Preprocessing Includes:
  Lowercasing
  Removing URLs, mentions, hashtags 
  Removing punctuation and digits
  Trimming extra spaces
  
⚙️ Setup Instructions
1. Clone the Repository
   git clone https://github.com/yourusername/airline-sentiment-analyzer.git
   cd airline-sentiment-analyzer
2.Create a Virtual Environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
   pip install -r requirements.txt
4.Download SpaCy Model
   python -m spacy download en_core_web_sm
5.Run the App
   streamlit run app.py

📦 Requirements
Your requirements.txt should include:
streamlit
scikit-learn
pandas
nltk
spacy

🚀 Deployment:
This app is deployed using Streamlit Community Cloud. To deploy your own version:
Push this project to a public GitHub repository
Go to streamlit.io/cloud
Sign in with GitHub
Click New App and select your repo
Set the file path to app.py
Click Deploy


