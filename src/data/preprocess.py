"""Text preprocessing for speech corpus."""
import re
import pandas as pd

BOILERPLATE_PATTERNS = [
    r"^(Mr\.\s*President|Madam\s*President|Mr\.\s*Chair|Madam\s*Chair)[,\.].*?[\.\n]",
    r"(I\s+thank|My\s+delegation\s+thanks|We\s+thank)\s+the\s+(delegation|representative|chair|president).*?[\.\n]",
    r"(I\s+congratulate|My\s+delegation\s+congratulates).*?[\.\n]",
    r"(Allow\s+me\s+to|Permit\s+me\s+to)\s+(associate|express|convey|begin).*?[\.\n]",
    r"\[.*?\]",  # bracketed content (translation notes, etc.)
]

def preprocess_text(text: str, remove_boilerplate: bool = True, lowercase: bool = False) -> str:
    """Clean speech text."""
    if not text:
        return ""

    result = text

    if remove_boilerplate:
        for pattern in BOILERPLATE_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)

    # Normalize whitespace
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    result = result.strip()

    if lowercase:
        result = result.lower()

    return result


def preprocess_corpus(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply preprocessing to corpus. Returns df with preprocessed text."""
    result = df.copy()
    result[text_col] = result[text_col].apply(
        lambda t: preprocess_text(str(t)) if pd.notna(t) else ""
    )
    return result
