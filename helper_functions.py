from bs4 import BeautifulSoup
import html
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# download NLTK resources
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def clean_text(text):
    # decode HTML entities
    text = html.unescape(text)
    # remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text().strip()

def preprocess_text(text):
    # remove newline characters
    text = text.replace("\n", " ") 
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # tokenize
    tokens = word_tokenize(text)
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)