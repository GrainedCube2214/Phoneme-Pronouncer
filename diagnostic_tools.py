# diagnostic_tools.py
"""
Diagnostic tools for analyzing pronunciation checker results:
1. Confusion matrix showing phoneme misclassifications
2. Distance distribution analysis
3. 2D/3D visualization of embeddings vs prototypes
4. Per-phoneme accuracy breakdown
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Phone families for analysis
PHONE_FAMILIES = {
    "stops": {"p","b","t","d","k","g"},
    "affricates": {"ch","jh"},
    "fricatives": {"f","v","th","dh","s","z","sh","zh","hh"},
    "nasals": {"m","n","ng"},
    "liquids": {"l","r","er"},
    "glides": {"w","y"},
    "vowels": {"iy","ih","eh","ey","ae","aa","ao","ah","uh","uw","ow","oy","aw","ay"},
}

def get_phone_family(phone):
    """Return the family name for a phoneme."""
    for family, phones in PHONE_FAMILIES.items():
        if phone in phones:
            return family
    return "other"


class PronunciationDiagnostics:
    def __init__(self, results_json_path, proto_pkl_path):
        """
        Args:
            results_json_path: Path to result.json from app.py
            proto_pkl_path: Path to phoneme_prototypes_tuned.pkl
        """
        with open(results_json_path, 'r') as f:
            self.results = json.load(f)
        
        with open(proto_pkl_path, 'rb') as f:
            self.protos = pickle.load(f)
        
        self.phone_results = self.results.get('per_phone_results', [])
        self.timeline = self.results.get('segments_timeline', [])
        
    def build_confusion_matrix(self, save_path='confusion_matrix.png'):
        """Build and visualize confusion matrix of target vs predicted phonemes."""
        # Collect all (target, predicted) pairs
        confusions = []
        for result in self.phone_results:
            if not result.get('known_target'):
                continue
            target = result.get('target')
            predicted = result.get('nearest_phoneme')
            confusions.append((target, predicted))
        
        if not confusions:
            print("No confusion data available")
            return
        
        # Get all unique phones
        all_phones = sorted(set([t for t, p in confusions] + [p for t, p in confusions]))
        
        # Build confusion matrix
        confusion_matrix = np.zeros((len(all_phones), len(all_phones)))
        phone_to_idx = {p: i for i, p in enumerate(all_phones)}
        
        for target, predicted in confusions:
            i = phone_to_idx[target]
            j = phone_to_idx[predicted]
            confusion_matrix[i, j] += 1
        
        # Plot confusion matrix
        plt.figure(figsize=(20, 18))
        
        # Normalize by row (show as percentages)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        confusion_pct = confusion_matrix / row_sums * 100
        
        # Create heatmap
        sns.heatmap(
            confusion_pct,
            xticklabels=all_phones,
            yticklabels=all_phones,
            annot=False,  # Too crowded with numbers
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'},
            square=True
        )
        
        plt.xlabel('Predicted Phoneme', fontsize=12)
        plt.ylabel('Target Phoneme', fontsize=12)
        plt.title('Confusion Matrix: Target vs Predicted Phonemes\n(Diagonal = Correct)', 
                  fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
        
        # Print top confusions
        print("\nTop 10 Confusions (target → predicted):")
        confusion_list = []
        for i, target in enumerate(all_phones):
            for j, predicted in enumerate(all_phones):
                if i != j and confusion_matrix[i, j] > 0:
                    confusion_list.append((
                        target, predicted, 
                        int(confusion_matrix[i, j]),
                        confusion_pct[i, j]
                    ))
        
        confusion_list.sort(key=lambda x: x[2], reverse=True)
        for target, pred, count, pct in confusion_list[:10]:
            target_fam = get_phone_family(target)
            pred_fam = get_phone_family(pred)
            same_fam = "✓" if target_fam == pred_fam else "✗"
            print(f"  {target:3s} → {pred:3s}: {count:2d} times ({pct:5.1f}%) "
                  f"[{target_fam:10s} → {pred_fam:10s}] {same_fam}")
    
    def analyze_distances(self, save_path='distance_analysis.png'):
        """Analyze distance distributions for correct vs incorrect predictions."""
        correct_dists = []
        incorrect_dists = []
        correct_margins = []
        incorrect_margins = []
        correct_confidence = []
        incorrect_confidence = []
        
        for result in self.phone_results:
            if not result.get('known_target'):
                continue
            
            dist = result.get('dist_to_target')
            margin = result.get('margin')
            conf = result.get('confidence', 0)
            is_correct = result.get('correct', False)
            
            if is_correct:
                correct_dists.append(dist)
                if margin is not None:
                    correct_margins.append(margin)
                correct_confidence.append(conf)
            else:
                incorrect_dists.append(dist)
                if margin is not None:
                    incorrect_margins.append(margin)
                incorrect_confidence.append(conf)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distance to target
        ax = axes[0, 0]
        if correct_dists:
            ax.hist(correct_dists, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
        if incorrect_dists:
            ax.hist(incorrect_dists, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        ax.set_xlabel('Distance to Target Prototype')
        ax.set_ylabel('Count')
        ax.set_title('Distance to Target: Correct vs Incorrect')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Margin (distance_other - distance_target)
        ax = axes[0, 1]
        if correct_margins:
            ax.hist(correct_margins, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
        if incorrect_margins:
            ax.hist(incorrect_margins, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero margin')
        ax.set_xlabel('Margin (dist_other - dist_target)')
        ax.set_ylabel('Count')
        ax.set_title('Margin Distribution: Correct vs Incorrect')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Confidence
        ax = axes[1, 0]
        if correct_confidence:
            ax.hist(correct_confidence, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
        if incorrect_confidence:
            ax.hist(incorrect_confidence, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        ax.axvline(0.25, color='orange', linestyle='--', linewidth=2, label='Threshold (0.25)')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution: Correct vs Incorrect')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Summary statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = "Summary Statistics\n" + "="*40 + "\n\n"
        stats_text += "Distance to Target:\n"
        if correct_dists:
            stats_text += f"  Correct:   μ={np.mean(correct_dists):.3f}, σ={np.std(correct_dists):.3f}\n"
        if incorrect_dists:
            stats_text += f"  Incorrect: μ={np.mean(incorrect_dists):.3f}, σ={np.std(incorrect_dists):.3f}\n"
        
        stats_text += "\nMargin:\n"
        if correct_margins:
            stats_text += f"  Correct:   μ={np.mean(correct_margins):.3f}, σ={np.std(correct_margins):.3f}\n"
        if incorrect_margins:
            stats_text += f"  Incorrect: μ={np.mean(incorrect_margins):.3f}, σ={np.std(incorrect_margins):.3f}\n"
        
        stats_text += "\nConfidence:\n"
        if correct_confidence:
            stats_text += f"  Correct:   μ={np.mean(correct_confidence):.3f}, σ={np.std(correct_confidence):.3f}\n"
        if incorrect_confidence:
            stats_text += f"  Incorrect: μ={np.mean(incorrect_confidence):.3f}, σ={np.std(incorrect_confidence):.3f}\n"
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distance analysis to {save_path}")
        plt.close()
    
    def per_phoneme_accuracy(self, save_path='per_phoneme_accuracy.png'):
        """Show accuracy breakdown by phoneme and phoneme family."""
        # Collect per-phoneme stats
        phone_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in self.phone_results:
            if not result.get('known_target'):
                continue
            target = result.get('target')
            is_correct = result.get('correct', False)
            
            phone_stats[target]['total'] += 1
            if is_correct:
                phone_stats[target]['correct'] += 1
        
        # Calculate accuracies
        phone_accuracies = []
        for phone in sorted(phone_stats.keys()):
            stats = phone_stats[phone]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            family = get_phone_family(phone)
            phone_accuracies.append({
                'phone': phone,
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total'],
                'family': family
            })
        
        df = pd.DataFrame(phone_accuracies)
        
        # Plot 1: Per-phoneme accuracy bar chart
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        ax = axes[0]
        colors = [plt.cm.RdYlGn(acc/100) for acc in df['accuracy']]
        bars = ax.bar(range(len(df)), df['accuracy'], color=colors, edgecolor='black')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['phone'], rotation=0, fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Phoneme Accuracy', fontsize=14, pad=10)
        ax.axhline(81.2, color='blue', linestyle='--', linewidth=2, label='Overall (81.2%)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # Add count labels on bars
        for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{row.correct}/{row.total}',
                   ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Family-level accuracy
        ax = axes[1]
        family_stats = df.groupby('family').agg({
            'correct': 'sum',
            'total': 'sum'
        })
        family_stats['accuracy'] = family_stats['correct'] / family_stats['total'] * 100
        family_stats = family_stats.sort_values('accuracy')
        
        colors = [plt.cm.RdYlGn(acc/100) for acc in family_stats['accuracy']]
        bars = ax.barh(range(len(family_stats)), family_stats['accuracy'], 
                       color=colors, edgecolor='black')
        ax.set_yticks(range(len(family_stats)))
        ax.set_yticklabels(family_stats.index, fontsize=11)
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Phoneme Family', fontsize=14, pad=10)
        ax.axvline(81.2, color='blue', linestyle='--', linewidth=2, label='Overall (81.2%)')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        # Add count labels
        for i, (bar, (family, row)) in enumerate(zip(bars, family_stats.iterrows())):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                   f'{int(row.correct)}/{int(row.total)}',
                   ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-phoneme accuracy to {save_path}")
        plt.close()
        
        return df
    
    def visualize_embedding_space_2d(self, result_json_path, embeddings_npz_path=None,
                                      save_path='embedding_space_2d.png'):
        """
        Visualize prototypes and actual segment embeddings in 2D using PCA.
        
        Args:
            result_json_path: Path to result.json
            embeddings_npz_path: Optional path to cached embeddings NPZ file
            save_path: Where to save the plot
        """
        # Get prototype centroids
        phones = sorted(self.protos.keys())
        centroids = np.array([self.protos[p]['centroid'] for p in phones])
        
        # Try to load segment embeddings
        if embeddings_npz_path and Path(embeddings_npz_path).exists():
            print(f"Loading embeddings from {embeddings_npz_path}")
            data = np.load(embeddings_npz_path)
            segment_embeddings = data['embeddings']
            segment_labels = data['labels'].tolist()
        else:
            print("No embeddings cache found. You'll need to generate one.")
            print("Run: python generate_embedding_cache.py <wav_path> <output.npz>")
            return
        
        # Combine all embeddings for PCA
        all_embeddings = np.vstack([centroids, segment_embeddings])
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        # Split back
        centroids_2d = embeddings_2d[:len(centroids)]
        segments_2d = embeddings_2d[len(centroids):]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        
        # Plot prototypes as large markers
        families = [get_phone_family(p) for p in phones]
        family_colors = {
            'vowels': 'red',
            'stops': 'blue',
            'fricatives': 'green',
            'nasals': 'purple',
            'liquids': 'orange',
            'glides': 'cyan',
            'affricates': 'magenta',
            'other': 'gray'
        }
        
        for i, (phone, family) in enumerate(zip(phones, families)):
            color = family_colors.get(family, 'gray')
            ax.scatter(centroids_2d[i, 0], centroids_2d[i, 1],
                      s=400, marker='*', c=color, edgecolors='black',
                      linewidths=2, alpha=0.8, zorder=10)
            ax.annotate(phone, (centroids_2d[i, 0], centroids_2d[i, 1]),
                       fontsize=10, fontweight='bold', ha='center', va='center')
        
        # Plot actual segments
        for i, label in enumerate(segment_labels):
            result = self.phone_results[i] if i < len(self.phone_results) else {}
            is_correct = result.get('correct', False)
            target_phone = result.get('target', label)
            
            # Find target centroid
            if target_phone in phones:
                target_idx = phones.index(target_phone)
                target_2d = centroids_2d[target_idx]
                
                # Draw line from segment to target
                color = 'green' if is_correct else 'red'
                alpha = 0.6 if is_correct else 0.8
                ax.plot([segments_2d[i, 0], target_2d[0]],
                       [segments_2d[i, 1], target_2d[1]],
                       color=color, alpha=alpha, linewidth=1, zorder=1)
                
                # Plot segment
                ax.scatter(segments_2d[i, 0], segments_2d[i, 1],
                          s=60, c=color, alpha=0.7, edgecolors='black',
                          linewidths=0.5, zorder=5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title('Phoneme Embedding Space (2D PCA Projection)\n'
                    'Stars = Prototypes, Circles = Actual Segments, Lines = Target Assignment',
                    fontsize=14, pad=15)
        ax.grid(alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Correct'),
            Patch(facecolor='red', alpha=0.7, label='Incorrect')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved 2D embedding visualization to {save_path}")
        plt.close()
    
    def generate_full_report(self, output_dir='diagnostics_output'):
        """Generate all diagnostic visualizations and save to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating diagnostic report...")
        print("="*60)
        
        self.build_confusion_matrix(output_path / 'confusion_matrix.png')
        self.analyze_distances(output_path / 'distance_analysis.png')
        df = self.per_phoneme_accuracy(output_path / 'per_phoneme_accuracy.png')
        
        # Save summary text report
        with open(output_path / 'summary_report.txt', 'w') as f:
            f.write("PRONUNCIATION CHECKER DIAGNOSTIC REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall Accuracy: {self.results.get('accuracy_over_known', 0)*100:.1f}%\n")
            f.write(f"Segments Evaluated: {self.results.get('segments_evaluated', 0)}\n")
            f.write(f"Known Targets: {self.results.get('n_known', 0)}\n")
            f.write(f"Correct: {self.results.get('n_correct', 0)}\n\n")
            
            diag = self.results.get('diagnostics', {})
            f.write(f"Inside Radius: {diag.get('inside_radius_frac', 0)*100:.1f}%\n")
            
            conf_stats = diag.get('confidence_stats', {})
            f.write(f"\nConfidence Statistics:\n")
            f.write(f"  Median: {conf_stats.get('median', 0):.3f}\n")
            f.write(f"  Mean: {conf_stats.get('mean', 0):.3f}\n")
            f.write(f"  Min: {conf_stats.get('min', 0):.3f}\n")
            f.write(f"  P90: {conf_stats.get('p90', 0):.3f}\n")
            
            f.write(f"\n\nPer-Phoneme Breakdown:\n")
            f.write("="*60 + "\n")
            for _, row in df.iterrows():
                f.write(f"{row['phone']:3s}: {row['accuracy']:5.1f}% ({row['correct']}/{row['total']}) [{row['family']}]\n")
        
        print(f"\nAll diagnostics saved to: {output_path}/")
        print("Files generated:")
        print("  - confusion_matrix.png")
        print("  - distance_analysis.png")
        print("  - per_phoneme_accuracy.png")
        print("  - summary_report.txt")


# Standalone script for generating embedding cache
EMBEDDING_CACHE_SCRIPT = """
# generate_embedding_cache.py
'''Generate embedding cache for visualization'''
import sys
import numpy as np
from models_runtime.asr_runtime import load_wav_16k
from models_runtime.embedding_runtime import SegmentEmbedder
from models_runtime.align_runtime import align_phones_mfa
from app import normalize_phone, MIN_SEG_MS

if len(sys.argv) < 3:
    print("Usage: python generate_embedding_cache.py <wav_path> <output.npz>")
    sys.exit(1)

wav_path = sys.argv[1]
output_path = sys.argv[2]

# Load and process
from models_runtime.asr_runtime import transcribe_word
text = transcribe_word(wav_path)
audio, sr = load_wav_16k(wav_path)

segments = align_phones_mfa(wav_path, text, target_word=None)

# Cut segments
seg_waves = []
seg_labels = []
min_len = int((MIN_SEG_MS / 1000.0) * sr)

for seg in segments:
    norm = normalize_phone(seg.phoneme)
    if not norm:
        continue
    
    start_idx = int(seg.start_s * sr)
    end_idx = int(seg.end_s * sr)
    if end_idx - start_idx < min_len:
        continue
    
    w = audio[start_idx:end_idx].astype("float32")
    seg_waves.append(w)
    seg_labels.append(norm)

# Embed
embedder = SegmentEmbedder()
embeddings = embedder.embed_segments(seg_waves)

# Save
np.savez(output_path, embeddings=embeddings, labels=np.array(seg_labels))
print(f"Saved {len(embeddings)} embeddings to {output_path}")
"""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate pronunciation diagnostics')
    parser.add_argument('result_json', help='Path to result.json from app.py')
    parser.add_argument('--proto-pkl', default='artifacts/phoneme_prototypes_tuned.pkl',
                       help='Path to prototype pickle file')
    parser.add_argument('--embeddings-npz', help='Optional: path to cached embeddings NPZ')
    parser.add_argument('--output-dir', default='diagnostics_output',
                       help='Output directory for diagnostic files')
    
    args = parser.parse_args()
    
    diag = PronunciationDiagnostics(args.result_json, args.proto_pkl)
    diag.generate_full_report(args.output_dir)
    
    # If embeddings provided, generate 2D visualization
    if args.embeddings_npz:
        diag.visualize_embedding_space_2d(
            args.result_json,
            args.embeddings_npz,
            Path(args.output_dir) / 'embedding_space_2d.png'
        )