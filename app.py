import streamlit as st
import joblib
import re

#========== LOAD THE MODEL YOU SAVED ===========
model = joblib.load('models/sentiment_model.pkl')

# ====== CLEANING THE TEXT ======
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)            # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)        # keep letters & spaces
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text

# UI DESIGNING
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Predict whether a movie review is **Positive** or **Negative** using a trained ML model.")

# Taking the review from user 
user_input = st.text_area("✍️ Enter your movie review here:")

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned_input = clean_text(user_input)
        prediction = model.predict([cleaned_input])[0]
        
        if prediction.lower() == "positive":
            st.success("✅ Sentiment: **Positive** 😊")
        elif prediction.lower() == "negative":
            st.error("🚫 Sentiment: **Negative** 😞")
        else:
            st.info(f"🤔 Sentiment: {prediction}")
