"""Normalize typed text input."""
import re

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Normalize common math notation
    text = text.replace(" ^ ", "**").replace("^", "**")
    text = text.replace("  ", " ")
    text = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", text)
    text = re.sub(r"(\d)\s*x\*\*", r"\1x**", text)
    return text
