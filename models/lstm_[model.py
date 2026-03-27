import numpy as np
from tensorflow.keras.models import Sequential
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
        raise ValueError("LSTM model is not built. Call build_model() first.")

    y = np.array(y).astype("float32")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)


def predict(text_seq):
    global model
    if model is None:
        raise ValueError("LSTM model is not built/trained yet.")

    pred = model.predict(text_seq, verbose=0)
    return int(pred[0][0] >= 0.5)