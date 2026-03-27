from transformers import pipeline, set_seed
import random
import re

set_seed(42)

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

PHISHING_SUBJECTS = [
    "Important Notice Regarding Your Account",
    "Urgent Action Required",
    "Verify Your Account Immediately",
    "Security Alert on Your Profile",
    "Payment Confirmation Required"
]

LEGIT_SUBJECTS = [
    "Meeting Reminder",
    "Your Recent Notification",
    "Project Update",
    "Appointment Confirmation",
    "Weekly Team Update"
]

PHISHING_CTA = [
    "Please verify your account immediately using the secure link below.",
    "Click the link below to confirm your identity and avoid service interruption.",
    "Review your account details now to prevent suspension.",
    "Take action now to secure your account and restore full access."
]

LEGIT_CTA = [
    "Please let us know if you have any questions.",
    "Feel free to contact us if you need any further information.",
    "Thank you for your attention to this update.",
    "We appreciate your continued cooperation."
]

SIGNATURES = {
    "phishing": ["Security Team", "Support Desk", "Account Services", "Verification Department"],
    "legitimate": ["Operations Team", "Customer Support", "Admin Team", "Communications Office"]
}

ORG_NAMES = {
    "phishing": ["SecureMail Services", "Account Protection Center", "Cloud Access Desk", "Digital Security Office"],
    "legitimate": ["Acme Solutions", "City University Office", "Northwind Services", "Project Coordination Team"]
}


def clean_generated_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(tone, target, difficulty, email_type):
    style_map = {
        "easy": "simple and obvious",
        "medium": "moderately convincing",
        "advanced": "highly convincing and realistic"
    }

    tone_map = {
        "urgent": "urgent and action-oriented",
        "formal": "formal and professional",
        "friendly": "friendly and conversational"
    }

    return f"""
Generate one complete email only.

The email must be fully structured and realistic.

Email type: {email_type}
Tone: {tone_map.get(tone, tone)}
Target audience: {target}
Difficulty: {style_map.get(difficulty, difficulty)}

Strict format:
Subject: <subject line>

Dear {target},

<body paragraph 1>

<body paragraph 2>

<call to action if phishing, or informative closing if legitimate>

Regards,
<team name>
<organization name>

Requirements:
- 80 to 150 words
- realistic wording
- proper grammar
- no bullet points
- no explanations
- no notes
- output only the final email
"""


def build_fallback_email(tone, target, difficulty, label):
    email_type = "phishing" if label == 1 else "legitimate"

    subject = random.choice(PHISHING_SUBJECTS if label == 1 else LEGIT_SUBJECTS)
    closing = random.choice(PHISHING_CTA if label == 1 else LEGIT_CTA)
    team = random.choice(SIGNATURES[email_type])
    org = random.choice(ORG_NAMES[email_type])

    if label == 1:
        body = (
            f"Subject: {subject}\n\n"
            f"Dear {target},\n\n"
            f"We detected unusual activity associated with your account earlier today. "
            f"For security reasons, your access may be limited unless your information is reviewed. "
            f"Our monitoring team has flagged this issue for immediate attention.\n\n"
            f"This matter should be handled as soon as possible to avoid possible disruption of service. "
            f"{closing}\n\n"
            f"Regards,\n{team}\n{org}"
        )
    else:
        body = (
            f"Subject: {subject}\n\n"
            f"Dear {target},\n\n"
            f"This is a reminder regarding an update related to your recent activity and scheduled communication. "
            f"We wanted to keep you informed so you have the latest information available.\n\n"
            f"No urgent action is required at this time. {closing}\n\n"
            f"Regards,\n{team}\n{org}"
        )

    return body


def generate_email(tone, target, difficulty, email_type):
    if email_type == "mixed":
        label = random.choice([0, 1])
    else:
        label = 1 if email_type == "phishing" else 0

    chosen_type = "phishing" if label == 1 else "legitimate"
    prompt = build_prompt(tone, target, difficulty, chosen_type)

    try:
        output = generator(
            prompt,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            truncation=True
        )[0]["generated_text"]

        generated_email = output.replace(prompt, "").strip()
        generated_email = clean_generated_text(generated_email)

        if "Subject:" not in generated_email or "Regards" not in generated_email:
            generated_email = build_fallback_email(tone, target, difficulty, label)

    except Exception:
        generated_email = build_fallback_email(tone, target, difficulty, label)

    return {
        "email": generated_email,
        "label": label,
        "difficulty": difficulty,
        "tone": tone,
        "target": target
    }


def generate_bulk(n, tone, target, difficulty, email_type):
    return [generate_email(tone, target, difficulty, email_type) for _ in range(n)]