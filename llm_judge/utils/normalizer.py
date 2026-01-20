# llm_judge/utils/normalizer.py
def normalize_label(text: str) -> str:
    text = text.strip().lower()

    if "positif" in text:
        return "positif"
    if "negatif" in text:
        return "negatif"
    if "netral" in text:
        return "netral"

    return "netral"
