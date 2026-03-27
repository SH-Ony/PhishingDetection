import re
from collections import Counter

TRIGGERS_LIST = [
    "urgent", "verify", "click", "password", "account", "confirm",
    "login", "reset", "immediately", "suspend", "security", "update"
]

SUSPICIOUS_PHRASES = [
    "click here",
    "verify your account",
    "reset your password",
    "confirm your identity",
    "login now",
    "action required",
    "account suspended",
    "security alert",
    "update your account",
    "limited access"
]

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "your", "you", "we", "our", "this", "that", "it", "be", "as", "at", "by",
    "from", "are", "was", "will", "can", "have", "has", "had", "not", "if",
    "please", "regards", "dear", "team", "hello", "thanks", "thank"
}


def extract_keywords(text, top_n=5):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
    filtered_words = [w for w in words if w not in STOPWORDS]
    word_freq = Counter(filtered_words)
    return [word for word, _ in word_freq.most_common(top_n)]


def detect_triggers(text):
    text_lower = str(text).lower()

    word_triggers = []
    for trigger in TRIGGERS_LIST:
        if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
            word_triggers.append(trigger)

    phrase_triggers = [phrase for phrase in SUSPICIOUS_PHRASES if phrase in text_lower]

    return sorted(list(set(word_triggers + phrase_triggers)))


def stylometric_features(text):
    text = str(text)
    words = text.split()

    return {
        "length": len(text),
        "word_count": len(words),
        "uppercase": sum(1 for c in text if c.isupper()),
        "exclamation_count": text.count("!"),
        "digit_count": sum(1 for c in text if c.isdigit()),
        "url_like_count": len(re.findall(r"http[s]?://|www\.|login|verify|update", text.lower()))
    }


def generate_reason(triggers, features):
    reasons = []

    if triggers:
        reasons.append("Presence of phishing-related trigger words or phrases")

    if features["uppercase"] > 20:
        reasons.append("High uppercase usage may indicate urgency or pressure")

    if features["exclamation_count"] > 2:
        reasons.append("Multiple exclamation marks suggest urgency or emotional pressure")

    if features["url_like_count"] > 0:
        reasons.append("Contains URL-like or action-oriented wording")

    if features["word_count"] < 20:
        reasons.append("Short message with limited context may be suspicious")

    if not reasons:
        reasons.append("No strong phishing indicators found")

    return "; ".join(reasons)


def analyze(text):
    text = str(text)

    triggers = detect_triggers(text)
    keywords = extract_keywords(text, top_n=5)
    features = stylometric_features(text)
    reason = generate_reason(triggers, features)

    return {
        "triggers": triggers,
        "keywords": keywords,
        "reason": reason,
        "length": features["length"],
        "word_count": features["word_count"],
        "uppercase": features["uppercase"],
        "exclamation_count": features["exclamation_count"],
        "digit_count": features["digit_count"],
        "url_like_count": features["url_like_count"]
    }