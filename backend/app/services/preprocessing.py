import re
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()
