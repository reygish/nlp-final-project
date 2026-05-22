# Symptom to Disease Classifier

A Streamlit web application that uses a Naive Bayes classifier to predict diseases based on symptom descriptions.

## Project Overview

This project implements:
- **NLP Preprocessing**: Tokenization, stop word removal, lemmatization
- **Machine Learning**: Naive Bayes classifier trained on symptom-disease pairs
- **Web Interface**: Interactive Streamlit application for easy user interaction

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data
Run Python and execute the following commands (one-time setup):
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
```

Or run the setup script if available.

## Running the Application

### Start the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. Enter your symptoms in the text area (e.g., "I have fever, cough, and sore throat")
2. Click the "Predict Disease" button
3. View the predicted disease and confidence score
4. **Important**: This is an AI-based tool for informational purposes. Always consult with a healthcare professional for medical advice.

## Files

- **model.pkl**: Trained Naive Bayes classifier
- **Symptom2Disease.csv**: Training data with symptoms and disease labels
- **solution.ipynb**: Original notebook with model training code
- **streamlit_app.py**: Streamlit web application
- **requirements.txt**: Python dependencies

## Features

- 🏥 Clean, user-friendly interface
- 📊 Real-time disease prediction
- 🎯 Confidence scores for predictions
- 💾 Model caching for fast predictions
- ⚠️ Medical disclaimer included

## Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for accurate medical guidance.
