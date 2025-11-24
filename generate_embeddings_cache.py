# generate_embedding_cache.py
"""
Generate embedding cache for visualization.
This extracts the actual embeddings from a WAV file along with their phone labels.
"""
import sys
import numpy as np
from pathlib import Path

from models_runtime.asr_runtime import transcribe_word, load_wav_16k
from models_runtime.embedding_runtime import SegmentEmbedder
from models_runtime.align_runtime import align_phones_mfa, PhonemeSegment

# Import from app.py
from app import normalize_phone, MIN_SEG_MS, IGNORE_PHONES, TIMIT_TO_CANON


def generate_embedding_cache(wav_path: str, output_path: str):
    """
    Generate embedding cache for a given WAV file.
    
    Args:
        wav_path: Path to audio file
        output_path: Where to save the NPZ file
    """
    print(f"Processing: {wav_path}")
    
    # 1) ASR
    text = transcribe_word(wav_path)
    print(f"Transcript: {text}")
    
    # 2) Load audio
    audio, sr = load_wav_16k(wav_path)
    
    # 3) Align
    segments = align_phones_mfa(wav_path, text, target_word=None)
    print(f"Got {len(segments)} segments from alignment")
    
    # 4) Cut and normalize segments
    seg_waves = []
    seg_labels = []
    seg_times = []
    min_len = int((MIN_SEG_MS / 1000.0) * sr)
    
    for seg in segments:
        norm = normalize_phone(seg.phoneme)
        if not norm:
            continue
        
        start_idx = int(seg.start_s * sr)
        end_idx = int(seg.end_s * sr)
        start_idx = max(0, start_idx)
        end_idx = min(len(audio), end_idx)
        
        if end_idx - start_idx < min_len:
            continue
        
        w = audio[start_idx:end_idx].astype("float32")
        seg_waves.append(w)
        seg_labels.append(norm)
        seg_times.append((seg.start_s, seg.end_s))
    
    print(f"Kept {len(seg_waves)} segments after filtering")
    
    # 5) Embed
    print("Generating embeddings...")
    embedder = SegmentEmbedder()
    embeddings = embedder.embed_segments(seg_waves)
    
    # 6) Save
    np.savez(
        output_path,
        embeddings=embeddings,
        labels=np.array(seg_labels),
        times=np.array(seg_times),
        wav_path=wav_path,
        transcript=text
    )
    
    print(f"✓ Saved {len(embeddings)} embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Phones: {' '.join(seg_labels)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_embedding_cache.py <wav_path> [output.npz]")
        print("\nExample:")
        print("  python generate_embedding_cache.py data_new/TRAIN/DR4/MMDM0/SI1311.WAV embeddings_cache.npz")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    
    # Auto-generate output name if not provided
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base = Path(wav_path).stem
        output_path = f"embeddings_{base}.npz"
    
    generate_embedding_cache(wav_path, output_path)