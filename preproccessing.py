import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global objects so training and inference use the same fitted objects
tfidf = TfidfVectorizer(max_features=5000)
tokenizer = Tokenizer(num_words=5000)

MAXLEN = 100


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ---------------- TF-IDF ----------------
def fit_tfidf(texts):
    return tfidf.fit_transform(texts)


def transform_tfidf(texts):
    return tfidf.transform(texts)


# ---------------- LSTM ----------------
def fit_lstm_tokenizer(texts):
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAXLEN)


def transform_lstm_sequences(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAXLEN)