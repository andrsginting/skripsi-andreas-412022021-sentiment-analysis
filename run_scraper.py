# run_scraper.py
# ======================================================
# Entry point utama untuk scraping multi-video YouTube
# ======================================================

import os
import time
from selenium.webdriver.common.by import By
from scrapping.driver import create_driver_visible
from scrapping.scraper import scrape_all_comments_batched


def get_video_title(driver):
    """Mengambil judul video dari elemen <h1> YouTube."""
    try:
        el = driver.find_element(By.XPATH, "//h1[@class='style-scope ytd-watch-metadata']")
        return el.text.strip()
    except Exception:
        return "Judul tidak ditemukan"


def run():
    print("Masukkan hingga 6 URL video YouTube (satu per baris).")
    print("Ketik 'selesai' untuk berhenti memasukkan URL.\n")

    # Ambil input semua URL di awal
    video_urls = []
    for i in range(6):
        url = input(f"URL Video ke-{i+1}: ").strip()
        if url.lower() == "selesai" or not url:
            break
        if not (url.startswith("https://www.youtube.com/watch?v=") or url.startswith("https://youtu.be/")):
            print("⚠️  URL tidak valid. Masukkan link YouTube yang benar.")
            continue
        video_urls.append(url)

    if not video_urls:
        print("Tidak ada URL yang dimasukkan. Program dihentikan.")
        return

    # 🔧 Pilihan apakah user ingin menampilkan browser atau tidak
    print("\nApakah Anda ingin menampilkan browser selama proses scraping?")
    print("Ketik 'y' untuk YA (tampilkan browser), atau 'n' untuk TIDAK (headless mode, lebih cepat).")
    show_browser = input("Pilihan Anda [y/n]: ").strip().lower() == 'y'

    if show_browser:
        print("\nMode aktif: 🖥️  Browser akan ditampilkan selama proses scraping.\n")
    else:
        print("\nMode aktif: ⚡ Headless mode — proses berjalan lebih cepat tanpa menampilkan browser.\n")

    print(f"Memulai scraping untuk {len(video_urls)} video...\n")
    time.sleep(1.2)

    # Jalankan scraping satu per satu video
    for idx, url in enumerate(video_urls, start=1):
        print("=" * 70)
        print(f"[VIDEO {idx}/{len(video_urls)}] Memproses: {url}")
        print("=" * 70)

        # Buat driver sesuai pilihan user
        driver = create_driver_visible(headless=not show_browser)
        driver.get(url)
        time.sleep(2.5)

        # Ambil dan tampilkan judul video
        video_title = get_video_title(driver)
        print(f"[INFO] Halaman video terdeteksi (judul video muncul).")
        print(f"[INFO] Judul video: {video_title}")

        # Jalankan proses scraping (batch per 10 komentar)
        csv_path, scraped_count, displayed_total = scrape_all_comments_batched(
            driver, url, batch_size=10, save_prefix=f"dataset_video_{idx}"
        )

        # Tutup browser setelah video selesai
        try:
            driver.quit()
        except Exception:
            pass

        print(f"\n[FINISH] Video #{idx} -> Data disimpan ke: {csv_path}")
        print(f"[RESULT] Total komentar di halaman: {displayed_total}")
        print(f"[RESULT] Total komentar di-scrape: {scraped_count}")
        if displayed_total:
            diff = displayed_total - scraped_count
            print(f"[CHECK] Selisih: {diff}")
            if abs(diff) <= 5:
                print("[OK] Akurat — jumlah hampir sama dengan tampilan YouTube ✅")
            else:
                print("[WARN] Terdapat perbedaan signifikan antara YouTube dan hasil scraping.")

        print("-" * 70)
        time.sleep(2)

    print("\n" + "=" * 70)
    print(f"✅ Semua {len(video_urls)} video telah selesai di-scrape.")
    print("=" * 70)


if __name__ == "__main__":
    run()
