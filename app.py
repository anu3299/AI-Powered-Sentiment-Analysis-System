
import os
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Find the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "notebbok",
    "models",
    "sentiment_model.pkl"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

# Application interface
st.title("AI-Powered Sentiment Analysis System")
st.write(
    "Analyze customer reviews using Machine Learning "
    "to predict Positive, Negative, or Neutral sentiment."
)

try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.info(
        "Check that sentiment_model.pkl exists inside "
        "notebbok/models."
    )
    st.stop()

# User input
review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: The product quality is excellent!",
    height=150
)

# Prediction
if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a customer review.")
    else:
        try:
            prediction = str(model.predict([review])[0])

            if prediction.lower() == "positive":
                st.success(f"Predicted Sentiment: {prediction}")
                st.write("The review expresses a positive opinion.")
            elif prediction.lower() == "negative":
                st.error(f"Predicted Sentiment: {prediction}")
                st.write("The review expresses a negative opinion.")
            else:
                st.info(f"Predicted Sentiment: {prediction}")
                st.write("The review expresses a neutral opinion.")

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))

st.divider()
st.caption(
    "Developed using Python, Scikit-learn, TF-IDF and Streamlit."
)
