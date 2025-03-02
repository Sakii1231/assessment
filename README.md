# News Topic Classification API

This project provides a FastAPI-based REST API designed to classify news articles into four categories: World, Sports, Business, and Sci/Tech. The classification is performed using a Support Vector Machine (SVM) model.

In the notebook, I trained multiple models including Naive Bayes, Logistic Regression, SVM, Random Forest, and FastText, and selected SVM as it performed the best.

## Running the API

### Running Without Docker

To run the API without Docker, follow these steps:

1. **Create and Activate a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies:**

```bash
pip install -r api_requirements.txt
```

3. **Run the FastAPI Application:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```


### Running With Docker

To run the API using Docker, simply execute:

```bash
docker-compose up
```

The API will be accessible at `http://localhost:8000`.

## API Endpoints

### POST /predict

This endpoint classifies a news article text into one of the four categories.

**Request Body:**

```json
{
    "text": "Your news article text here"
}
```

**Response:**

```json
{
    "class": 0,
    "class_name": "World"
}
```


## Running the Notebook

To work with the notebook, follow these steps:

1. **Install Notebook Requirements:**

```bash
pip install -r notebook_requirements.txt
```

2. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```


## Text Processing Pipeline

The API employs a two-step text processing pipeline:

1. **Text Cleaning:**
    - Removes HTML tags.
    - Decodes HTML entities.
2. **Text Preprocessing:**
    - Converts text to lowercase.
    - Removes punctuation.
    - Tokenizes the text.
    - Removes stopwords.
    - Applies lemmatization.

## Model Details

The API utilizes the following components:

- **Feature Extraction:** TF-IDF vectorizer.
- **Classifier:** SVM for prediction.
- **Classification Categories:** World, Sports, Business, and Sci/Tech.
- **Model Files:** The models are loaded from pickle files located at `svm_model_files/`, specifically `svm_model.pkl` and `tfidf_vectorizer.pkl`.


## Error Handling

The API includes robust error handling with appropriate HTTP status codes and error messages for various scenarios.