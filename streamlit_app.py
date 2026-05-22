import streamlit as st
import pickle
import pandas as pd
from string import punctuation
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

st.set_page_config(
    page_title="Symptom to Disease Classifier",
    page_icon=":hospital:",
    layout="centered"
)

@st.cache_resource
def load_resources():
    """Load the model and frequency distribution"""
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        df = pd.read_csv('Symptom2Disease.csv')
        sentences = "  ".join(df['text'])
        cleaned_tokens = preprocess(sentences)
        fd = FreqDist(cleaned_tokens)
        
        return model, fd
    except FileNotFoundError:
        st.error("Model or data file not found!")
        return None, None

def tag_to_wordnet(tag: str):
    """Convert NLTK pos tag to WordNet tag"""
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('R'):
        return wordnet.ADV
    if tag.startswith('N'):
        return wordnet.NOUN
    return wordnet.NOUN

def preprocess(sentences: str):
    """Preprocess text: tokenize, remove stopwords, lemmatize"""
    tokens = word_tokenize(sentences)
    cleaned_tokens = [
        token.lower() 
        for token in tokens 
        if token.lower() not in stopwords.words('english') 
        and token.lower() not in punctuation
    ]
    tagged_tokens = pos_tag(cleaned_tokens)
    stemmed_tokens = [
        WordNetLemmatizer().lemmatize(token, tag_to_wordnet(tag)) 
        for token, tag in tagged_tokens
    ]
    return stemmed_tokens

def feature_extraction(tokens: list[str], fd: FreqDist):
    """Extract features from tokens based on frequency distribution"""
    dict_feature = {}
    for token in fd.keys():
        dict_feature[token] = (token in tokens)
    return dict_feature

def predict_disease(symptom_text: str, model, fd: FreqDist):
    """Predict disease from symptom text"""
    if not symptom_text.strip():
        return None, None
    
    tokens = preprocess(symptom_text)
    features = feature_extraction(tokens, fd)
    
    prediction = model.classify(features)
    probabilities = model.prob_classify(features)
    confidence = probabilities.prob(prediction)
    
    return prediction, confidence

model, fd = load_resources()

if model is not None and fd is not None:
    st.title(":hospital: Symptom to Disease Classifier")
    st.markdown("""
    Enter your symptoms to get a disease prediction powered by Natural Language Processing and Machine Learning.
    """)
    
    st.subheader("Enter Your Symptoms")
    
    symptom_input = st.text_area(
        "Describe your symptoms (e.g., 'I have fever, cough, and sore throat')",
        placeholder="Type your symptoms here...",
        height=120
    )
    
    if st.button("Predict Disease", type="primary"):
        if symptom_input.strip():
            with st.spinner("Analyzing symptoms..."):
                prediction, confidence = predict_disease(symptom_input, model, fd)
                
                if prediction:
                    st.success("Prediction Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Disease", prediction)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    st.warning(
                        "⚠️ **Disclaimer**: This is an AI-based prediction tool and should not be used as a substitute for professional medical advice. "
                        "Please consult with a healthcare professional for accurate diagnosis and treatment."
                    )
        else:
            st.warning("Please enter some symptoms to proceed.")
    
    with st.expander("📋 Example Symptoms"):
        st.markdown("""
        - "allergy"
        - "mallaria"
        - "hypertension"
        - "common cold"
        """)
else:
    st.error("Failed to load the model. Please ensure 'model.pkl' and 'Symptom2Disease.csv' are in the same directory.")
