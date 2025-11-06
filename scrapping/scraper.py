import os
import time
import random
import csv
import re
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .utils import extract_comment_richtext


from .driver import create_driver_visible
from .utils import parse_numeric_text, make_hash_id, clean_comment_text_preserve

# ==========================================================
# Config
# ==========================================================
BATCH_SIZE = 10
INITIAL_SCROLLS = (400, 800, 1200)
SCROLL_PAUSE = (0.5, 1.0)
MAX_GLOBAL_SCROLLS = 300
STABLE_CHECKS = 5
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)


# ==========================================================
# Helpers
# ==========================================================
def safe_click(driver, element):
    """Klik elemen dengan fallback ke JavaScript."""
    try:
        element.click()
        return True
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False


def _wait_for_video_loaded(driver, timeout=40):
    """Tunggu sampai judul video muncul di halaman."""
    print("[INFO] Menunggu halaman video dimuat sepenuhnya...")
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.XPATH, "//h1[@class='style-scope ytd-watch-metadata']"))
    )
    print("[INFO] Halaman video terdeteksi (judul video muncul).")


def _scroll_until_comments_area(driver, max_tries=20):
    """
    Scroll bertahap sampai area komentar dan header komentar (jumlah komentar) benar-benar terlihat.
    Fase 1: cari ytd-comments (memicu modul komentar)
    Fase 2: scroll tambahan agar count muncul di viewport
    """
    print("[INFO] Mencari dan menampilkan area komentar YouTube...")
    found_comments = False

    # --- Fase 1: pastikan modul komentar sudah dimuat ---
    for i in range(max_tries):
        driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(0.9 + random.random() * 0.4)
        try:
            comments_area = driver.find_element(By.TAG_NAME, "ytd-comments")
            if comments_area.is_displayed():
                found_comments = True
                print(f"[INFO] Area komentar ditemukan setelah {i+1} kali scroll.")
                break
        except Exception:
            continue

    if not found_comments:
        print("[WARN] Area komentar belum ditemukan setelah scroll maksimal.")
        return

    # --- Fase 2: scroll lebih dalam agar header count masuk viewport ---
    print("[INFO] Scroll tambahan untuk memastikan header komentar terlihat...")
    for j in range(8):
        driver.execute_script("window.scrollBy(0, 300);")
        time.sleep(0.8 + random.random() * 0.3)
        try:
            header_el = driver.find_element(
                By.XPATH, "//ytd-comments-header-renderer//h2[@id='count']"
            )
            if header_el.is_displayed():
                print(f"[INFO] Header komentar terlihat setelah tambahan scroll ke-{j+1}.")
                break
        except Exception:
            continue


def _wait_for_comment_count(driver, timeout=25):
    """Tunggu hingga jumlah komentar muncul di halaman YouTube (struktur baru/lama)."""
    print("[INFO] Menunggu teks jumlah komentar muncul...")

    # Pola XPath untuk berbagai struktur DOM
    patterns = [
        # Struktur terbaru (h2 > yt-formatted-string > span:first-child)
        "//ytd-comments-header-renderer//h2[@id='count']//yt-formatted-string//span[1]",
        # Struktur lama
        "//ytd-comments-header-renderer//yt-formatted-string[@id='count']",
        # Struktur campuran
        "//ytd-comments-header-renderer//yt-formatted-string[contains(@class,'count-text')]//span[1]",
    ]

    for xp in patterns:
        try:
            el = WebDriverWait(driver, timeout).until(
                EC.visibility_of_element_located((By.XPATH, xp))
            )
            txt = el.text.strip()
            if txt:
                m = re.search(r'([\d,\.]+)', txt.replace('\xa0', ' '))
                if m:
                    total = int(m.group(1).replace(',', '').split('.')[0])
                    print(f"[INFO] Jumlah komentar di halaman YouTube: {total}")
                    return total
        except Exception:
            continue

    # === Fallback: gunakan JS query seluruh header ===
    try:
        txt = driver.execute_script("""
            let el = document.querySelector('ytd-comments-header-renderer');
            return el ? el.innerText : '';
        """)
        if txt:
            m = re.search(r'([\d,\.]+)', txt.replace('\xa0', ' '))
            if m:
                total = int(m.group(1).replace(',', '').split('.')[0])
                print(f"[INFO] Jumlah komentar di halaman YouTube (via JS): {total}")
                return total
    except Exception:
        pass

    print("[WARN] Tidak bisa membaca jumlah komentar meskipun header komentar terlihat.")
    return None


def _continuous_scroll_until_stable(driver, max_scrolls=MAX_GLOBAL_SCROLLS, stable_checks=STABLE_CHECKS):
    """Scroll sampai tidak ada thread baru yang termuat."""
    last_count = 0
    stable = 0
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(random.uniform(*SCROLL_PAUSE))
        threads_now = driver.find_elements(By.XPATH, "//ytd-comment-thread-renderer")
        cur = len(threads_now)
        if cur == last_count:
            stable += 1
        else:
            stable = 0
            last_count = cur
        if stable >= stable_checks:
            break
    return last_count


# ==========================================================
# Deep Recursive Reply Expansion
# ==========================================================
def expand_replies_recursively(thread, driver, csv_writer, processed_hash, depth=0, max_depth=50):
    """
    Rekursif membuka semua 'Show more replies' dan 'Read more' di balasan komentar.
    Kini kompatibel dengan struktur baru (2025) dan lama.
    """
    if depth > max_depth:
        print(f"[DEBUG] Depth {depth} melebihi batas, hentikan rekursi.")
        return

    time.sleep(0.1 + random.random() * 0.05)

    # Ambil semua balasan di dalam thread
    replies = thread.find_elements(By.XPATH, ".//ytd-comment-replies-renderer//ytd-comment-view-model")

    # Buka 'Read more' di tiap balasan
    for r in replies:
        read_btns = r.find_elements(By.XPATH, ".//tp-yt-paper-button[@id='more']")
        for btn in read_btns:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                if safe_click(driver, btn):
                    time.sleep(0.08 + random.random() * 0.05)
            except Exception:
                continue

        # Simpan komentar balasan ke CSV
        try:
            r_element = r.find_element(By.ID, "content-text")
            r_text = clean_comment_text_preserve(extract_comment_richtext(r_element))
            r_likes_text = ""
            try:
                r_likes_text = r.find_element(By.ID, "vote-count-middle").text.strip()
            except Exception:
                pass
            r_likes = parse_numeric_text(r_likes_text)
            r_cid = make_hash_id(r_text)
            if r_cid not in processed_hash:
                csv_writer.writerow({"comment": r_text, "likes_count": r_likes, "is_reply": True})
                processed_hash.add(r_cid)
        except Exception:
            continue

    # Cari tombol "Show more replies" di struktur baru maupun lama
    button_selectors = [
        # Struktur baru 2025
        ".//ytd-continuation-item-renderer//button[contains(., 'Show more replies')]",
        # Struktur lama
        ".//ytd-button-renderer[@id='more-replies' or @id='more-replies-sub-thread']//button",
    ]
    buttons = []
    for sel in button_selectors:
        buttons += thread.find_elements(By.XPATH, sel)

    # Klik tombol jika ada
    if not buttons:
        return

    for btn in buttons:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            label = btn.get_attribute("aria-label") or btn.text
            if any(k in label for k in ["Show more replies", "Tampilkan", "Lihat", "Mostrar"]):
                if safe_click(driver, btn):
                    print(f"[DEBUG] membuka 'Show more replies' (depth={depth})")
                    time.sleep(0.3 + random.random() * 0.2)
                    expand_replies_recursively(thread, driver, csv_writer, processed_hash, depth + 1)
        except Exception:
            continue



# ==========================================================
# Process one comment thread (fully expanded)
# ==========================================================
def process_thread_fully(thread, driver, csv_writer, processed_hash):
    """Proses satu komentar utama + semua balasannya (depth-first)."""
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", thread)
    time.sleep(0.15)

    # Buka “Read more” pada komentar utama
    for btn in thread.find_elements(By.XPATH, ".//tp-yt-paper-button[@id='more']"):
        safe_click(driver, btn)
        time.sleep(0.05)

    # Ambil komentar utama
    try:
        main = thread.find_element(By.XPATH, ".//ytd-comment-view-model[@id='comment']")
        main_element = main.find_element(By.ID, "content-text")
        main_text = clean_comment_text_preserve(extract_comment_richtext(main_element))
        likes_text = ""
        try:
            likes_text = main.find_element(By.ID, "vote-count-middle").text.strip()
        except Exception:
            pass
        likes = parse_numeric_text(likes_text)
        cid = make_hash_id(main_text)
        if cid not in processed_hash:
            csv_writer.writerow({"comment": main_text, "likes_count": likes, "is_reply": False})
            processed_hash.add(cid)
    except Exception:
        return

    # Klik “View replies” pertama kali
    first_buttons = thread.find_elements(By.XPATH, ".//ytd-button-renderer[@id='more-replies']")
    for btn in first_buttons:
        safe_click(driver, btn)
        time.sleep(0.25)

    # Jalankan recursive expansion untuk memastikan semua balasan habis
    time.sleep(0.3)
    expand_replies_recursively(thread, driver, csv_writer, processed_hash, depth=1)


# ==========================================================
# Batch-level scraping
# ==========================================================
def scrape_in_batches(driver, batch_size, csv_path):
    """Scrape komentar dalam batch dengan progress bar."""
    threads = driver.find_elements(By.XPATH, "//ytd-comment-thread-renderer")
    total_threads = len(threads)
    n_batches = (total_threads + batch_size - 1) // batch_size

    f = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=["comment", "likes_count", "is_reply"], quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    processed_hash = set()
    scraped_total = 0

    for batch_index in range(n_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, total_threads)
        batch_threads = threads[start:end]

        print(f"\n[BATCH {batch_index+1}/{n_batches}] Memproses {len(batch_threads)} komentar utama...")
        start_time = time.time()

        for thr in tqdm(batch_threads, desc=f" Batch {batch_index+1}/{n_batches}", ncols=80):
            try:
                process_thread_fully(thr, driver, writer, processed_hash)
            except Exception:
                continue
            time.sleep(0.07 + random.random() * 0.05)

        elapsed = time.time() - start_time
        scraped_total = len(processed_hash)
        print(f"[TIME] Durasi batch {batch_index+1}: {elapsed:.1f} detik")
        print(f"[DONE] Batch {batch_index+1}/{n_batches} selesai. Total komentar unik: {scraped_total}")
        print("-" * 70)

    f.close()
    return scraped_total


# ==========================================================
# Entry point
# ==========================================================
def scrape_all_comments_batched(driver, video_url, batch_size=BATCH_SIZE, save_prefix="dataset_video"):
    """Main entry point untuk scraping 1 video YouTube."""
    csv_path = os.path.join(DATASET_DIR, f"{save_prefix}.csv")

    print(f"\n[START] Scraping video: {video_url}")
    driver.get(video_url)

    # ✅ Tunggu video utama termuat
    try:
        _wait_for_video_loaded(driver)
    except Exception:
        print("[WARN] Tidak mendeteksi judul video dalam waktu wajar, lanjut ke proses scroll.")

    # ✅ Scroll perlahan sampai area komentar muncul
    _scroll_until_comments_area(driver)

    # ✅ Pastikan jumlah komentar terbaca sebelum lanjut
    displayed_total = _wait_for_comment_count(driver, timeout=30)

    # ✅ Tunggu komentar pertama muncul
    wait = WebDriverWait(driver, 25)
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "(//ytd-comment-view-model)[1]")))
        print("[INFO] Komentar pertama muncul.")
    except Exception:
        print("[ERROR] Tidak ada komentar muncul (timeout).")
        return csv_path, 0, displayed_total

    # ✅ Scroll sampai semua thread dimuat
    total_threads = _continuous_scroll_until_stable(driver)
    print(f"[INFO] Jumlah thread komentar termuat: {total_threads}")

    # ✅ Jalankan batching
    scraped_total = scrape_in_batches(driver, batch_size, csv_path)

    print(f"\n[SUMMARY] Komentar di YouTube: {displayed_total if displayed_total else 'Tidak terbaca'}")
    print(f"[SUMMARY] Komentar berhasil di-scrape: {scraped_total}")
    if displayed_total:
        diff = displayed_total - scraped_total
        if abs(diff) <= 5:
            print(f"[OK] Selisih kecil ({diff}) — hasil sangat akurat ✅")
        else:
            print(f"[WARN] Selisih {diff} komentar (kemungkinan komentar disembunyikan).")

    return csv_path, scraped_total, displayed_total
