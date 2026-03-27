import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_FEATURES = 5000
MAXLEN = 100

tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
tokenizer = Tokenizer(num_words=MAX_FEATURES)


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def fit_tfidf(texts):
    return tfidf.fit_transform(texts)


def transform_tfidf(texts):
    return tfidf.transform(texts)


def fit_lstm_tokenizer(texts):
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAXLEN)


def transform_lstm_sequences(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAXLEN)


def save_preprocessors(base_path="saved_models"):
    joblib.dump(tfidf, f"{base_path}/tfidf_vectorizer.pkl")
    joblib.dump(tokenizer, f"{base_path}/tokenizer.pkl")


def load_preprocessors(base_path="saved_models"):
    global tfidf, tokenizer
    tfidf = joblib.load(f"{base_path}/tfidf_vectorizer.pkl")
    tokenizer = joblib.load(f"{base_path}/tokenizer.pkl")