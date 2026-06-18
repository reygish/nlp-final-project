import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from string import punctuation
from gensim.models import Word2Vec
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')

st.set_page_config(
    page_title="Symptom to Disease Classifier",
    page_icon=":hospital:",
    layout="centered"
)

st.title(":hospital: Symptom to Disease Classifier")
st.markdown(
    "Enter your symptoms to get a disease prediction powered by Natural Language Processing and Machine Learning."
)

@st.cache_resource
def load_bow_resources():
    """Load Bag-of-Words model, DictVectorizer, and frequency distribution."""
    model_path = os.path.join("models", "bow_model.joblib")
    vectorizer_path = os.path.join("models", "bow_vectorizer.joblib")

    if not os.path.exists(model_path):
        st.error("Missing Bag-of-Words model file: models/bow_model.joblib")
        return None, None, None

    if not os.path.exists(vectorizer_path):
        st.error("Missing Bag-of-Words vectorizer file: models/bow_vectorizer.joblib")
        return None, None, None

    model = joblib.load(model_path)
    dv = joblib.load(vectorizer_path)

    df = pd.read_csv("Symptom2Disease.csv")
    sentences = "  ".join(df["text"])
    cleaned_tokens = preprocess_tokens(sentences)
    fd = FreqDist(cleaned_tokens)

    return model, dv, fd

@st.cache_resource
def load_tfidf_resources():
    """Load TF-IDF model and vectorizer (fallback to fit if missing)."""
    model_path = os.path.join("models", "tfidf_model.joblib")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")

    if not os.path.exists(model_path):
        st.error("Missing TF-IDF model file: models/tfidf_model.joblib")
        return None, None

    model = joblib.load(model_path)

    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        df = pd.read_csv("Symptom2Disease.csv")
        vectorizer = TfidfVectorizer()
        vectorizer.fit(df["text"].astype(str).apply(preprocess_text))

    return model, vectorizer

@st.cache_resource
def load_word2vec_resources():
    """Load Word2Vec classifier and embeddings model."""
    model_path = os.path.join("models", "word2vec_model.joblib")
    vectors_path = os.path.join("models", "word2vec_vectors.model")

    if not os.path.exists(model_path):
        st.error("Missing Word2Vec model file: models/word2vec_model.joblib")
        return None, None

    if not os.path.exists(vectors_path):
        st.error("Missing Word2Vec vectors file: models/word2vec_vectors.model")
        return None, None

    model = joblib.load(model_path)
    w2v_model = Word2Vec.load(vectors_path)
    return model, w2v_model

@st.cache_resource
def load_glove_resources():
    """Load GloVe classifier and embeddings from the glove file."""
    model_path = os.path.join("models", "glove_model.joblib")
    glove_path = os.path.join("glove.6B", "glove.6B.100d.txt")

    if not os.path.exists(model_path):
        st.error("Missing GloVe model file: models/glove_model.joblib")
        return None, None

    if not os.path.exists(glove_path):
        st.error("Missing GloVe embeddings file: glove.6B/glove.6B.100d.txt")
        return None, None

    model = joblib.load(model_path)
    glove_embeddings = load_glove_embeddings(glove_path, vector_size=100)
    return model, glove_embeddings

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

def preprocess_tokens(sentences: str):
    """Preprocess text: tokenize, remove stopwords, lemmatize."""
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

def preprocess_text(sentences: str):
    """Preprocess text and return a whitespace-joined string."""
    return " ".join(preprocess_tokens(sentences))

def feature_extraction(tokens: list[str], fd: FreqDist):
    """Extract features from tokens based on frequency distribution"""
    dict_feature = {}
    for token in fd.keys():
        dict_feature[token] = (token in tokens)
    return dict_feature

def sentence_vector(tokens, embeddings, vector_size):
    vectors = []
    for word in tokens:
        if word in embeddings:
            vectors.append(embeddings[word])
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def load_glove_embeddings(path, vector_size=100):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != vector_size + 1:
                continue
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def predict_nltk(symptom_text: str, model, dv, fd: FreqDist):
    if not symptom_text.strip():
        return None, None

    tokens = preprocess_tokens(symptom_text)
    features = feature_extraction(tokens, fd)

    features_vec = dv.transform([features])
    prediction = model.predict(features_vec)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features_vec)))
    else:
        confidence = None

    return prediction, confidence

def predict_tfidf(symptom_text: str, model, vectorizer):
    if not symptom_text.strip():
        return None, None

    processed_text = preprocess_text(symptom_text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)))
    else:
        confidence = None

    return prediction, confidence

def predict_word2vec(symptom_text: str, model, w2v_model):
    if not symptom_text.strip():
        return None, None

    tokens = preprocess_tokens(symptom_text)
    vector = sentence_vector(tokens, w2v_model.wv, w2v_model.wv.vector_size)
    prediction = model.predict([vector])[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba([vector])))
    else:
        confidence = None

    return prediction, confidence

def predict_glove(symptom_text: str, model, glove_embeddings, vector_size=300):
    if not symptom_text.strip():
        return None, None

    tokens = preprocess_tokens(symptom_text)
    vector = sentence_vector(tokens, glove_embeddings, vector_size)
    prediction = model.predict([vector])[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba([vector])))
    else:
        confidence = None

    return prediction, confidence

model_choice = st.radio(
    "Choose a model:",
    ["Bag-of-Words", "TF-IDF", "Word2Vec", "GloVe"],
    horizontal=True
)

resources = None
if model_choice == "Bag-of-Words":
    resources = load_bow_resources()
elif model_choice == "TF-IDF":
    resources = load_tfidf_resources()
elif model_choice == "Word2Vec":
    resources = load_word2vec_resources()
elif model_choice == "GloVe":
    resources = load_glove_resources()

if resources and all(item is not None for item in resources):
    
    st.subheader("Enter Your Symptoms")
    
    symptom_input = st.text_area(
        "Describe your symptoms (e.g., 'I have fever, cough, and sore throat')",
        placeholder="Type your symptoms here...",
        value="I've been sneezing a lot and my nose feels so clogged. I can't even smell anything!",
        height=120
    )
    
    if st.button("Predict Disease", type="primary"):
        if symptom_input.strip():
            with st.spinner("Analyzing symptoms..."):
                if model_choice == "Bag-of-Words":
                    model, dv, fd = resources
                    prediction, confidence = predict_nltk(symptom_input, model, dv, fd)
                elif model_choice == "TF-IDF":
                    model, vectorizer = resources
                    prediction, confidence = predict_tfidf(symptom_input, model, vectorizer)
                elif model_choice == "Word2Vec":
                    model, w2v_model = resources
                    prediction, confidence = predict_word2vec(symptom_input, model, w2v_model)
                else:
                    model, glove_embeddings = resources
                    prediction, confidence = predict_glove(symptom_input, model, glove_embeddings, vector_size=100)
                
                if prediction:
                    st.success("Prediction Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Disease", prediction)
                    
                    with col2:
                        if confidence is not None:
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        else:
                            st.metric("Confidence", "N/A")
                    
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
    st.error("Failed to load the selected model and required resources.")
