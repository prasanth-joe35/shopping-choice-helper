import re

MAX_LEN = 1800

_BAD_PHRASES = re.compile(
    r"(?is)(?:ignore.*instruction|reveal.*prompt|system\s*prompt|"
    r"disregard.*previous|override.*guard|developer\s*notes)"
)

_PII = [
    r"\b\d{10}\b",  # 10-digit phone
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
]

def sanitize(text: str) -> str:
    text = (text or "")[:MAX_LEN]
    text = _BAD_PHRASES.sub("", text)
    # Strip URLs to prevent prompt stuffing via links
    text = re.sub(r"https?://\S+", "[LINK]", text)
    return text.strip()

def redact_pii(text: str) -> str:
    safe = text or ""
    for pat in _PII:
        safe = re.sub(pat, "[REDACTED]", safe)
    return safe
