import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = None


def build_model(vocab_size=5000, embedding_dim=64, input_length=100):
    global model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )


def train_model(X, y, epochs=3, batch_size=32):
    global model
    if model is None:
        raise ValueError("LSTM model is not built.")
    y = np.array(y).astype("float32")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)


def predict(text_seq):
    global model
    if model is None:
        raise ValueError("LSTM model is not loaded or trained.")
    pred = model.predict(text_seq, verbose=0)
    return int(pred[0][0] >= 0.5)


def save_model_file(base_path="saved_models"):
    global model
    os.makedirs(base_path, exist_ok=True)
    model.save(f"{base_path}/lstm_model.keras")


def load_model_file(base_path="saved_models"):
    global model
    model = load_model(f"{base_path}/lstm_model.keras")