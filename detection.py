from models.ml_model import predict_all_models
from models.lstm_model import predict as lstm_predict
from models.bert_model import predict as bert_predict


def majority_vote(preds):
    return int(sum(preds) >= (len(preds) / 2))


def detect(email, tfidf_vec, lstm_seq):
    ml_preds = predict_all_models(tfidf_vec)
    lstm_pred = lstm_predict(lstm_seq)
    bert_pred = bert_predict(email)

    all_preds = list(ml_preds.values()) + [lstm_pred, bert_pred]
    final_pred = majority_vote(all_preds)

    return {
        "ml_models": ml_preds,
        "lstm": lstm_pred,
        "bert": bert_pred,
        "final": final_pred
    }