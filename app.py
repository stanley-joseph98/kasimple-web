from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

# Define request model
class EmailText(BaseModel):
    text: str

# Load pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = FastAPI()

stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

@app.post("/predict")
def predict(email: EmailText):
    text = preprocess(email.text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    label = "Spam" if prediction[0] == 1 else "Ham"
    return {"label": label}
