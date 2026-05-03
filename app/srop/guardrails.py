"""
E5: Input guardrails.
- Refuse out-of-scope queries.
- Redact PII from logs.
"""
import re

from app.srop.errors import GuardrailRefusalError

OUT_OF_SCOPE_PATTERNS = [
    r"\bwrite me a poem\b",
    r"\btell me a joke\b",
    r"\bwrite.*code for\b",
    r"\bgenerate.*image\b",
    r"\bplay a game\b",
]

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "[CC_REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    (r"\b\d{10}\b", "[PHONE_REDACTED]"),
]


def check_guardrails(message: str) -> None:
    """Raise GuardrailRefusalError if message is out of scope."""
    lower = message.lower()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lower):
            raise GuardrailRefusalError(
                "I can only assist with Helix product questions and account management."
            )


def redact_pii(text: str) -> str:
    """Redact PII from text before logging."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
