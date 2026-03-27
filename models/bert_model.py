PHISHING_KEYWORDS = [
    "urgent", "verify", "click", "password", "account", "confirm",
    "suspend", "suspension", "login", "limited access", "security alert",
    "reset", "immediately", "action required", "bank", "payment",
    "update your account", "validate"
]


def predict(text):
    text = str(text).lower()
    score = sum(1 for keyword in PHISHING_KEYWORDS if keyword in text)
    return 1 if score >= 2 else 0