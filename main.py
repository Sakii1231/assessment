from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from helper_functions import clean_text, preprocess_text

# Create a FastAPI application
app = FastAPI()

# Load the SVM model and TfidfVectorizer
svm = joblib.load('svm_model_files/svm_model.pkl')
tfidf_vectorizer = joblib.load('svm_model_files/tfidf_vectorizer.pkl')

# Dictionary to map class numbers to topic names
class_dict = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech'
}

# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    text: str

# Define the POST endpoint
@app.post("/predict", response_model=dict)
async def predict_news_topic(request: TextRequest):
    try:
        # Convert the text to a list and transform it using TfidfVectorizer
        data = request.text
        data = clean_text(data)
        data = preprocess_text(data)
        text_data = [data]
        vectorized_text = tfidf_vectorizer.transform(text_data)

        # Make a prediction using the SVM model
        prediction = svm.predict(vectorized_text)

        # Get the class name from the dictionary
        class_name = class_dict[prediction[0]]

        # Return the prediction as a JSON object
        return {
            "class": int(prediction[0]),
            "class_name": class_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
