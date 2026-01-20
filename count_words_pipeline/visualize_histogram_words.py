import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

class HistogramWordCountVisualizer:
    """Visualize word count distribution from YouTube comments"""
    
    def __init__(self, data_dir=None):
        """Initialize visualizer with data directory"""
        if data_dir is None:
            # Assume script is in count_words_pipeline folder
            self.data_dir = Path(__file__).parent.parent / "dataset_count_word"
        else:
            self.data_dir = Path(data_dir)
        
        self.csv_files = []
        self.all_word_counts = []
        
    def load_data(self):
        """Load word counts from all CSV files"""
        print(f"Loading data from: {self.data_dir}")
        
        # Find all cleaned CSV files
        csv_pattern = self.data_dir / "count_words_dataset_video_*_cleaned.csv"
        self.csv_files = sorted(self.data_dir.glob("count_words_dataset_video_*_cleaned.csv"))
        
        if not self.csv_files:
            print(f"Warning: No CSV files found in {self.data_dir}")
            return False
        
        print(f"Found {len(self.csv_files)} files")
        
        # Load word counts from each file
        for csv_file in self.csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'word_count' in df.columns:
                    self.all_word_counts.extend(df['word_count'].tolist())
                    print(f"  ✓ {csv_file.name}: {len(df)} comments loaded")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")
        
        print(f"\nTotal comments loaded: {len(self.all_word_counts)}")
        return len(self.all_word_counts) > 0
    
    def create_histogram(self, output_path=None, show_stats=True):
        """Create and display histogram of word counts"""
        if not self.all_word_counts:
            print("No data to visualize. Please load data first.")
            return
        
        # Create figure with larger size
        plt.figure(figsize=(14, 8))
        
        # Create histogram with appropriate bins
        word_counts = np.array(self.all_word_counts)
        bins = self._determine_optimal_bins(word_counts)
        
        counts, edges, patches = plt.hist(
            word_counts,
            bins=bins,
            color='#2E86AB',
            edgecolor='black',
            alpha=0.7,
            linewidth=1.2
        )
        
        # Customize plot
        plt.xlabel('Jumlah Kata per Komentar', fontsize=13, fontweight='bold')
        plt.ylabel('Jumlah Komentar (Frekuensi)', fontsize=13, fontweight='bold')
        plt.title('Distribusi Panjang Komentar YouTube\nKarakteristik Teks Komentar Secara Umum', 
                  fontsize=15, fontweight='bold', pad=20)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Add statistics box
        if show_stats:
            stats_text = self._get_statistics_text(word_counts)
            plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            output_path = self.data_dir.parent / "histogram_word_count.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Histogram saved to: {output_path}")
        
        return output_path
    
    def _determine_optimal_bins(self, data):
        """Determine optimal number of bins using Sturges' formula"""
        n = len(data)
        max_val = int(np.max(data))
        
        # Use Sturges formula: k = 1 + log2(n)
        sturges_bins = int(np.ceil(np.log2(n) + 1))
        
        # But also consider the range - use smaller of the two
        bins = min(sturges_bins, max_val // 2 + 1)
        
        # Ensure at least 20 bins for granularity
        bins = max(20, bins)
        
        return bins
    
    def _get_statistics_text(self, word_counts):
        """Generate statistics text for the plot"""
        stats = {
            'Mean': np.mean(word_counts),
            'Median': np.median(word_counts),
            'Std Dev': np.std(word_counts),
            'Min': np.min(word_counts),
            'Max': np.max(word_counts),
            'Total': len(word_counts)
        }
        
        text = "Statistik Distribusi:\n"
        text += f"Mean: {stats['Mean']:.2f}\n"
        text += f"Median: {stats['Median']:.2f}\n"
        text += f"Std Dev: {stats['Std Dev']:.2f}\n"
        text += f"Min: {stats['Min']:.0f}\n"
        text += f"Max: {stats['Max']:.0f}\n"
        text += f"Total: {stats['Total']:.0f}"
        
        return text
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.all_word_counts:
            print("No data loaded")
            return
        
        word_counts = np.array(self.all_word_counts)
        
        print("\n" + "="*60)
        print("RINGKASAN DISTRIBUSI PANJANG KOMENTAR")
        print("="*60)
        print(f"Total Komentar: {len(word_counts)}")
        print(f"Rata-rata kata/komentar: {np.mean(word_counts):.2f}")
        print(f"Median: {np.median(word_counts):.2f}")
        print(f"Standar Deviasi: {np.std(word_counts):.2f}")
        print(f"Min kata: {np.min(word_counts)}")
        print(f"Max kata: {np.max(word_counts)}")
        print("-"*60)
        
        # Distribution percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print("Persentil Distribusi:")
        for p in percentiles:
            value = np.percentile(word_counts, p)
            print(f"  {p}th percentile: {value:.0f} kata")
        print("="*60 + "\n")


def main():
    """Main execution function"""
    # Create visualizer instance
    visualizer = HistogramWordCountVisualizer()
    
    # Load data
    if not visualizer.load_data():
        print("Failed to load data")
        return
    
    # Print summary statistics
    visualizer.print_summary()
    
    # Create and save histogram
    visualizer.create_histogram()
    
    print("✓ Visualisasi histogram berhasil dibuat!")


if __name__ == "__main__":
    main()
