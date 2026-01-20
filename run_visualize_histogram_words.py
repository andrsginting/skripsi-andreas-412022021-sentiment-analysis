#!/usr/bin/env python3
"""
Runner script untuk visualisasi histogram distribusi panjang komentar
Jalankan dari folder project-skripsi (root)
"""

import sys
from pathlib import Path

# Add count_words_pipeline to Python path
pipeline_dir = Path(__file__).parent / "count_words_pipeline"
sys.path.insert(0, str(pipeline_dir))

# Import visualizer
from visualize_histogram_words import HistogramWordCountVisualizer

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("VISUALISASI HISTOGRAM DISTRIBUSI PANJANG KOMENTAR YOUTUBE")
    print("="*70 + "\n")
    
    # Define data directory
    data_dir = Path(__file__).parent / "dataset_count_word"
    
    # Create visualizer
    visualizer = HistogramWordCountVisualizer(data_dir=data_dir)
    
    # Load data
    print("📂 Memuat data dari dataset_count_word...")
    if not visualizer.load_data():
        print("❌ Gagal memuat data")
        return 1
    
    # Print statistics
    visualizer.print_summary()
    
    # Create histogram
    print("📊 Membuat histogram...")
    output_file = visualizer.create_histogram()
    
    print(f"\n✅ Selesai!")
    print(f"📍 Output histogram: {output_file}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())
