# scrapping/utils.py
# GAP 1 — Imports & Helper Functions

import re, hashlib
from bs4 import BeautifulSoup


def parse_numeric_text(s):
    """Konversi teks likes seperti '1.2K' ke integer. Robust terhadap empty string."""
    if not s:
        return 0
    s = str(s).strip().lower().replace(',', '')
    try:
        if 'k' in s:
            return int(float(s.replace('k','')) * 1_000)
        elif 'm' in s:
            return int(float(s.replace('m','')) * 1_000_000)
        else:
            # extract digits only
            digits = re.sub(r'[^0-9]', '', s)
            return int(digits) if digits else 0
    except Exception:
        return 0

def make_hash_id(text):
    """Buat hash unik dari text (untuk deduplikasi cepat)."""
    if text is None:
        text = ""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_comment_text_preserve(text):
    """
    Membersihkan text agar tetap 1 baris tanpa menghapus emoji, URL, atau simbol khusus.
    - Hanya menghapus newline dan carriage return
    - Menjaga karakter non-ASCII (emoji, simbol)
    - Tidak menyentuh karakter tanda baca, URL, atau emotikon
    """
    if text is None:
        return ""

    # Pastikan text adalah string murni
    s = str(text)

    # Ganti newline dan carriage return dengan spasi
    s = s.replace('\r', ' ').replace('\n', ' ')

    # Hapus spasi ganda atau spasi berlebihan
    s = re.sub(r'\s+', ' ', s).strip()

    # Kembalikan teks tanpa menyentuh emoji, simbol, atau URL
    return s


def extract_comment_richtext(element):
    """
    Mengambil isi komentar YouTube dengan:
    ✅ Menyertakan emoji (dari <img alt="❤">)
    ✅ Menyertakan hyperlink eksternal (https://...) dan video YouTube
    🚫 Tidak menambahkan URL untuk mention akun (@username)
    """
    try:
        html = element.get_attribute("innerHTML")
        soup = BeautifulSoup(html, "html.parser")

        # Ganti semua emoji <img alt="❤"> → ❤
        for img in soup.find_all("img"):
            if img.has_attr("alt"):
                img.replace_with(img["alt"])
            elif img.has_attr("src"):
                img.replace_with("[emoji]")

        # Ganti semua link <a href="...">
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text(" ", strip=True)

            # 1️⃣ Jika mention akun (/@username) → tampilkan teks saja
            if href.startswith("/@"):
                a.replace_with(text)
                continue

            # 2️⃣ Jika link internal YouTube video/channel lain
            elif href.startswith("/watch") or href.startswith("/channel"):
                href_full = "https://www.youtube.com" + href
                a.replace_with(f"{text} ({href_full})" if text else href_full)

            # 3️⃣ Jika link eksternal (http/https)
            elif href.startswith("http"):
                a.replace_with(f"{text} ({href})" if text else href)

            # 4️⃣ Jika tidak dikenali → ambil teksnya saja
            else:
                a.replace_with(text)

        # Ambil hasil teks penuh
        result = soup.get_text(" ", strip=True)
        result = re.sub(r"\s+", " ", result)
        return result.strip()
    except Exception:
        try:
            return element.text.strip()
        except Exception:
            return ""

