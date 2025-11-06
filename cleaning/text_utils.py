# cleaning/text_utils.py
import re
import emoji

def remove_mentions_hashtags(text: str) -> str:
    """Menghapus @mention dan #hashtag dari teks."""
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    return text

def remove_urls(text: str) -> str:
    """Menghapus semua URL dan tanda kurung yang membungkusnya."""
    # hapus URL dalam kurung (contoh: (https://...))
    text = re.sub(r"\(\s*https?://[^\)]*\)", "", text)
    # hapus URL berdiri sendiri
    text = re.sub(r"https?://\S+", "", text)
    return text

def remove_emoticons(text: str) -> str:
    """Menghapus emoji & emoticon unicode."""
    text = emoji.replace_emoji(text, replace="")
    # hapus karakter non-alfanumerik kecuali tanda baca dasar
    text = re.sub(r"[^\w\s.,!?'\"]+", " ", text)
    return text

def remove_outer_quotes(text: str) -> str:
    """Menghapus tanda petik di awal/akhir (termasuk ganda)."""
    text = text.strip()
    text = re.sub(r'^[\'"]+|[\'"]+$', "", text)
    text = text.replace('""', '"').strip('"').strip("'")
    return text.strip()

def remove_punctuation_at_start(text: str) -> str:
    """Menghapus tanda baca di awal komentar."""
    return re.sub(r'^[\.,;:!?\'"()\[\]\-]+', '', text).strip()

def clean_spacing(text: str) -> str:
    """Merapikan spasi berlebih."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_lowercase(text: str) -> str:
    """Mengubah semua huruf ke lowercase."""
    return text.lower()

def clean_comment_pipeline(text: str) -> str:
    """Pipeline penuh cleaning teks komentar sesuai instruksi user."""
    text = str(text)
    text = remove_mentions_hashtags(text)
    text = remove_urls(text)
    text = remove_emoticons(text)
    text = remove_outer_quotes(text)
    text = clean_spacing(text)
    text = remove_punctuation_at_start(text)
    text = normalize_lowercase(text)
    return text.strip()

# =======================================================
# HELPER: DETEKSI PENYEBAB DATA KOSONG
# =======================================================
def detect_empty_reason(original_text: str) -> str:
    """Mendeteksi penyebab komentar kosong setelah cleaning."""
    if not original_text or str(original_text).strip() == "":
        return "kosong_asli"
    if emoji.emoji_count(original_text) > 0:
        return "emoji_saja"
    if re.fullmatch(r"[\W_]+", str(original_text)):
        return "tanda_baca_saja"
    return "lainnya"
