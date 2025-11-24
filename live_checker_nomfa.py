#!/usr/bin/env python3
"""
Truly zero-shot live pronunciation checker.
Uses sliding window phoneme detection - no dictionary needed.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os

from models_runtime.asr_runtime import load_wav_16k
from models_runtime.embedding_runtime import SegmentEmbedder
from models_runtime.phonemespace_runtime import PhonemeSpace

SAMPLE_RATE = 16000
DURATION_SECONDS = 5

# Window-based segmentation
WINDOW_MS = 120  # 120ms window
HOP_MS = 40      # 40ms hop (overlapping windows)

class C:
    G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; B = '\033[94m'
    RESET = '\033[0m'; BOLD = '\033[1m'


def sliding_window_segments(audio, sr=SAMPLE_RATE, window_ms=WINDOW_MS, hop_ms=HOP_MS):
    """
    Create overlapping windows for phoneme detection.
    Each window is a potential phoneme.
    """
    window_samples = int(window_ms / 1000.0 * sr)
    hop_samples = int(hop_ms / 1000.0 * sr)
    
    segments = []
    
    for start_idx in range(0, len(audio) - window_samples, hop_samples):
        end_idx = start_idx + window_samples
        
        seg_audio = audio[start_idx:end_idx]
        start_s = start_idx / sr
        end_s = end_idx / sr
        
        # Check energy (skip silent segments)
        energy = np.mean(seg_audio ** 2)
        if energy < 0.001:  # Very quiet, skip
            continue
        
        segments.append({
            'audio': seg_audio,
            'start_s': start_s,
            'end_s': end_s,
            'energy': energy
        })
    
    return segments


def detect_phonemes_sliding_window(wav_path):
    """
    Zero-shot phoneme detection using sliding windows.
    No dictionary or MFA needed.
    """
    # Load audio
    audio, sr = load_wav_16k(wav_path)
    
    # Get ASR transcript (for display only, not used in detection)
    from models_runtime.asr_runtime import transcribe_word
    transcript = transcribe_word(wav_path)
    
    # Create sliding windows
    windows = sliding_window_segments(audio, sr)
    
    if not windows:
        return {
            'error': 'No speech detected in audio',
            'transcript': transcript
        }
    
    # Load models
    embedder = SegmentEmbedder()
    space = PhonemeSpace("artifacts/phoneme_prototypes_large.pkl")
    
    # Embed all windows
    window_audios = [w['audio'] for w in windows]
    embeddings = embedder.embed_segments(window_audios)
    
    # Classify each window
    detections = []
    
    for window, emb in zip(windows, embeddings):
        # Find nearest phoneme (no target - pure detection)
        cls = space.classify_embedding(emb, target_phone=None)
        
        nearest = cls.get('nearest_phoneme')
        dist = cls.get('dist_to_nearest')
        radius = cls.get('radius_nearest')
        confidence = 1.0 - (dist / radius) if radius > 0 else 0.0
        
        detections.append({
            'phoneme': nearest,
            'start': window['start_s'],
            'end': window['end_s'],
            'confidence': confidence,
            'distance': dist,
            'within_radius': dist <= radius
        })
    
    # Merge consecutive same-phoneme detections
    merged = []
    
    for det in detections:
        if merged and merged[-1]['phoneme'] == det['phoneme']:
            # Extend previous
            merged[-1]['end'] = det['end']
            merged[-1]['confidence'] = max(merged[-1]['confidence'], det['confidence'])
        else:
            # New phoneme
            merged.append(det.copy())
    
    # Filter very short detections (likely noise)
    MIN_DURATION = 0.03  # 30ms
    filtered = [m for m in merged if (m['end'] - m['start']) >= MIN_DURATION]
    
    # Compute overall stats
    avg_confidence = np.mean([d['confidence'] for d in filtered]) if filtered else 0
    high_conf_count = sum(1 for d in filtered if d['confidence'] > 0.3)
    
    return {
        'transcript': transcript,
        'detected_phonemes': filtered,
        'n_phonemes': len(filtered),
        'avg_confidence': avg_confidence,
        'high_confidence_count': high_conf_count,
        'phoneme_sequence': ' '.join(d['phoneme'] for d in filtered)
    }


def print_zeroshot_results(result):
    """Print results from zero-shot detection."""
    
    if 'error' in result:
        print(f"{C.R}❌ Error: {result['error']}{C.RESET}\n")
        return
    
    transcript = result['transcript']
    phonemes = result['detected_phonemes']
    avg_conf = result['avg_confidence']
    high_conf = result['high_confidence_count']
    n_total = result['n_phonemes']
    
    print(f"{C.BOLD}{'='*70}{C.RESET}")
    print(f"{C.BOLD}ZERO-SHOT PRONUNCIATION ANALYSIS{C.RESET}")
    print(f"{C.BOLD}{'='*70}{C.RESET}\n")
    
    print(f"{C.B}You said:{C.RESET} \"{transcript}\"\n")
    
    # Quality score
    quality_score = (high_conf / n_total * 100) if n_total > 0 else 0
    
    if quality_score >= 80:
        color, emoji = C.G, "🌟"
        msg = "Excellent clarity!"
    elif quality_score >= 60:
        color, emoji = C.Y, "👍"
        msg = "Good pronunciation"
    else:
        color, emoji = C.R, "📚"
        msg = "Practice more"
    
    print(f"{color}{emoji} Clarity Score: {quality_score:.0f}% ({high_conf}/{n_total} phonemes clear){C.RESET}")
    print(f"{color}{msg}{C.RESET}\n")
    
    # Show detected phonemes
    print(f"{C.BOLD}Detected Phonemes:{C.RESET}\n  ", end='')
    
    for i, det in enumerate(phonemes):
        phone = det['phoneme']
        conf = det['confidence']
        
        if conf > 0.3:
            print(f"{C.G}{phone:3s}{C.RESET}({conf:.2f}) ", end='')
        elif conf > 0.2:
            print(f"{C.Y}{phone:3s}{C.RESET}({conf:.2f}) ", end='')
        else:
            print(f"{C.R}{phone:3s}{C.RESET}({conf:.2f}) ", end='')
        
        if (i + 1) % 7 == 0:
            print("\n  ", end='')
    
    print(f"\n\n{C.Y}Phoneme sequence: {result['phoneme_sequence']}{C.RESET}\n")
    
    # Low confidence phonemes
    weak_phonemes = [d for d in phonemes if d['confidence'] < 0.25]
    
    if weak_phonemes:
        print(f"{C.R}{C.BOLD}Unclear phonemes (practice these):{C.RESET}")
        for w in weak_phonemes:
            print(f"  {C.R}'{w['phoneme']}'{C.RESET} at {w['start']:.1f}s (confidence: {w['confidence']:.2f})")
    else:
        print(f"{C.G}🎉 All phonemes were clear!{C.RESET}")
    
    print(f"\n{C.BOLD}{'='*70}{C.RESET}\n")


def record_audio(duration=DURATION_SECONDS):
    """Record with normalization."""
    print(f"\n{C.Y}🎤 Recording for {duration} seconds... SPEAK NOW!{C.RESET}\n")
    
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    audio = audio.flatten()
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.7
    
    print(f"{C.G}✓ Recording complete!{C.RESET}\n")
    return audio


def main():
    print(f"{C.BOLD}ZERO-SHOT PRONUNCIATION CHECKER{C.RESET}\n")
    print("How it works:")
    print("  • Listens to your speech")
    print("  • Automatically detects what you said")
    print("  • Identifies unclear phonemes")
    print("  • No manual input needed!\n")
    
    while True:
        cmd = input(f"{C.B}Press Enter to record (or 'quit'): {C.RESET}")
        
        if cmd.lower() in ['q', 'quit', 'exit']:
            print(f"\n{C.G}Goodbye!{C.RESET}")
            break
        
        # Record
        audio = record_audio()
        
        # Save
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(temp_wav, audio, SAMPLE_RATE)
        
        try:
            # Analyze
            print(f"{C.Y}Analyzing...{C.RESET}\n")
            result = detect_phonemes_sliding_window(temp_wav)
            
            # Display
            print_zeroshot_results(result)
            
        except Exception as e:
            print(f"{C.R}Error: {e}{C.RESET}\n")
        
        finally:
            os.remove(temp_wav)
        
        print(f"{C.Y}{'─'*70}{C.RESET}\n")


if __name__ == '__main__':
    main()