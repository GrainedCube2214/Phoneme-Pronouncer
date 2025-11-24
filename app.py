# app.py
import json
import argparse
import numpy as np

from models_runtime.phonemespace_runtime import PhonemeSpace
from models_runtime.asr_runtime import transcribe_word, load_wav_16k
from models_runtime.embedding_runtime import SegmentEmbedder, SAMPLE_RATE
from models_runtime.align_runtime import align_phones_mfa, PhonemeSegment

# -------------------------------
# Load P95 radii from JSON
# -------------------------------
with open("artifacts/phoneme_prototypes_large.json", "r") as f:  # was embeds_partial.json
    P95_DATA = json.load(f)
    P95_RADII = {ph: data["p95_radius"] for ph, data in P95_DATA.items()}


# -------------------------------
# (1) Phone normalization helpers
# -------------------------------
IGNORE_PHONES = {
    "sil", "sp", "spn", "pau", "epi", "h#", ""   # skip from scoring/timeline
}

TIMIT_TO_CANON = {
    # closures -> their stop (common in TIMIT/MFA outputs)
    "pcl": "p", "tcl": "t", "kcl": "k", "bcl": "b", "dcl": "d", "gcl": "g",
    # TIMIT-ish variants → CMU-ish
    "ix": "ih", "ax": "ah", "ax-h": "ah", "axr": "er",
    "el": "l", "em": "m", "en": "n", "nx": "n",
    "ux": "uw", "zh": "sh",
    # often collapsed in 39-phone sets
    "ao": "aa",
}

def normalize_phone(p: str) -> str:
    """Map aligner labels to your canonical set; return '' to ignore."""
    if p is None:
        return ""
    p = p.strip().lower()
    p = TIMIT_TO_CANON.get(p, p)
    if p in IGNORE_PHONES:
        return ""
    return p


# -------------------------------------------
# Decision rule parameters
# -------------------------------------------
USE_P95_RADII = True         # Use p95 radii from JSON instead of PKL
CONFIDENCE_THRESHOLD = 0.20   # from 0.25
CONFIDENCE_SAME_FAMILY = 0.12 # from 0.15

MIN_SEG_MS = 60              # discard segments shorter than 60 ms (stabilize embeddings)

# Phone families for lenient treatment of natural confusions
PHONE_FAMILIES = {
    "stops": {"p","b","t","d","k","g"},
    "affricates": {"ch","jh"},
    "fricatives": {"f","v","th","dh","s","z","sh","zh","hh"},
    "nasals": {"m","n","ng"},
    "liquids": {"l","r","er"},
    "glides": {"w","y"},
    "vowels": {"iy","ih","eh","ey","ae","aa","ao","ah","uh","uw","ow","oy","aw","ay","er"},
}

def same_family(a: str, b: str) -> bool:
    if not a or not b:
        return False
    for fam in PHONE_FAMILIES.values():
        if a in fam and b in fam:
            return True
    return False


class PronunciationChecker:
    def __init__(self,
                 proto_path: str,
                 encoder_name: str = "facebook/wav2vec2-large-960h"):  # was base-960h
        self.space = PhonemeSpace(proto_path)
        self.embedder = SegmentEmbedder(encoder_name)

    def analyze_file(self, wav_path: str):
        # 1) ASR – full intended text for the utterance
        text = transcribe_word(wav_path)
        words = text.split()
        if not words:
            return {"error": "ASR produced empty text", "raw_text": text}

        # 2) Load audio (16k mono)
        audio, sr = load_wav_16k(wav_path)   # np.ndarray, 16k mono

        # 3) Align phones with MFA over the FULL sentence (no clipping by target word)
        try:
            segments = align_phones_mfa(
                wav_path=wav_path,
                transcript=text,
                target_word=None,
                prefer_sidecar=False
            )
        except TypeError:
            segments = align_phones_mfa(
                wav_path=wav_path,
                transcript=text,
                target_word=None
            )

        if not segments:
            return {
                "error": "MFA produced no segments",
                "raw_text": text,
                "target_word": None,
            }

        # 4) Cut segments, normalize labels, and collect only usable ones
        seg_waves = []
        seg_targets = []
        kept_segments = []
        min_len = int((MIN_SEG_MS / 1000.0) * sr)

        for seg in segments:
            norm = normalize_phone(seg.phoneme)
            if not norm:
                continue

            start_idx = int(seg.start_s * sr)
            end_idx   = int(seg.end_s   * sr)
            start_idx = max(0, start_idx)
            end_idx   = min(len(audio), end_idx)
            if end_idx - start_idx < min_len:
                continue

            w = audio[start_idx:end_idx].astype("float32")
            seg_waves.append(w)
            seg_targets.append(norm)
            kept_segments.append(
                PhonemeSegment(norm, seg.start_s, seg.end_s, getattr(seg, "file_path", None))
            )

        if not seg_waves:
            return {
                "error": "No usable segments from alignment after normalization/length filters",
                "raw_text": text,
                "target_word": None,
            }

        # 5) Embed all segments
        embs = self.embedder.embed_segments(seg_waves)  # (B, D)

        # 6) Classify each segment with new decision rule
        results = []
        n_correct = 0
        n_known = 0

        for ph, z, seg in zip(seg_targets, embs, kept_segments):
            cls = self.space.classify_embedding(z, target_phone=ph)
            
            # Preserve original fields
            orig_correct = cls.get("correct")
            orig_radius_target = cls.get("radius_target")
            dist_tgt = cls.get("dist_to_target")
            dist_other = cls.get("dist_to_nearest_other")
            nearest_other = cls.get("nearest_other_phoneme")
            confidence = cls.get("confidence", 0.0)

            # NEW DECISION RULE
            if cls.get("known_target") and (dist_tgt is not None) and (orig_radius_target is not None):
                n_known += 1
                
                # Use p95 radius from JSON (95th percentile acceptance threshold)
                if USE_P95_RADII and ph in P95_RADII:
                    acceptance_radius = P95_RADII[ph]
                else:
                    # Fallback to PKL radius with scaling
                    acceptance_radius = float(orig_radius_target) * 1.1
                
                inside = (dist_tgt <= acceptance_radius)
                
                # Compute margin FIRST (before using it)
                actual_margin = None if dist_other is None else (dist_other - dist_tgt)

                # Decision: inside radius + confidence threshold
                if inside:
                    # Determine confidence threshold based on phoneme family
                    if nearest_other and same_family(ph, nearest_other):
                        threshold = CONFIDENCE_SAME_FAMILY
                    else:
                        threshold = CONFIDENCE_THRESHOLD
                    
                    adjusted_correct = (confidence >= threshold)
                else:
                    # Outside radius = incorrect
                    adjusted_correct = False
                    threshold = None

                # Update result fields
                cls["correct_old"] = orig_correct
                cls["radius_target_scaled"] = acceptance_radius
                cls["margin"] = actual_margin
                cls["confidence_threshold_used"] = threshold
                cls["decision_rule"] = {
                    "use_p95_radii": USE_P95_RADII,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "confidence_same_family": CONFIDENCE_SAME_FAMILY,
                    "same_family_detected": (nearest_other and same_family(ph, nearest_other)),
                }
                cls["correct"] = adjusted_correct

                if adjusted_correct:
                    n_correct += 1
            else:
                # Unknown target phones are not counted toward accuracy
                pass

            results.append(cls)

        # 7) Build timeline
        segments_timeline = []
        for seg, cls in zip(kept_segments, results):
            segments_timeline.append({
                "phoneme": seg.phoneme,
                "start": seg.start_s,
                "end": seg.end_s,
                "known_target": bool(cls.get("known_target")),
                "correct": (cls.get("correct") is True),
            })

        acc_known = (n_correct / n_known) if n_known > 0 else None
        coverage = (n_known / len(results)) if len(results) > 0 else None

        # Diagnostics
        inside_count = sum(
            1 for r in results
            if r.get("known_target") and
               (r.get("dist_to_target") is not None) and
               (r.get("radius_target_scaled") is not None) and
               (r["dist_to_target"] <= r["radius_target_scaled"])
        )
        
        margins = [
            (r.get("dist_to_nearest_other") - r.get("dist_to_target"))
            for r in results
            if r.get("known_target")
            and (r.get("dist_to_nearest_other") is not None)
            and (r.get("dist_to_target") is not None)
        ]
        
        confidences = [
            r.get("confidence", 0.0)
            for r in results
            if r.get("known_target")
        ]
        
        diag = {
            "inside_radius_frac": (inside_count / n_known) if n_known > 0 else None,
            "inside_count": inside_count,
            "outside_count": n_known - inside_count,
            "margin_stats": {
                "count": len(margins),
                "min": float(np.min(margins)) if margins else None,
                "median": float(np.median(margins)) if margins else None,
                "p90": float(np.percentile(margins, 90)) if margins else None
            },
            "confidence_stats": {
                "count": len(confidences),
                "min": float(np.min(confidences)) if confidences else None,
                "median": float(np.median(confidences)) if confidences else None,
                "mean": float(np.mean(confidences)) if confidences else None,
                "p90": float(np.percentile(confidences, 90)) if confidences else None
            }
        }

        return {
            "raw_text": text,
            "target_word": None,
            "canonical_phones": None,
            "segments_evaluated": len(results),
            "n_known": n_known,
            "n_correct": n_correct,
            "coverage_known": coverage,
            "accuracy_over_known": acc_known,
            "per_phone_results": results,
            "segments_timeline": segments_timeline,
            "diagnostics": diag,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="path/to/audio.wav")
    parser.add_argument("--out", "-o", default="result.json",
                        help="Where to write JSON output (default: result.json)")
    args = parser.parse_args()

    proto_path = "artifacts/phoneme_prototypes_large.pkl" # was _tuned.pkl
    checker = PronunciationChecker(proto_path=proto_path)
    result = checker.analyze_file(args.wav_path)

    with open(args.out, "w") as f:
    # Convert numpy types to Python native types
        json.dump(result, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"Wrote results to {args.out}")
    
    # Print summary
    if "accuracy_over_known" in result and result["accuracy_over_known"] is not None:
        print(f"\nAccuracy: {result['accuracy_over_known']:.1%} ({result['n_correct']}/{result['n_known']})")
        if "diagnostics" in result:
            diag = result["diagnostics"]
            if diag.get("inside_radius_frac") is not None:
                print(f"Inside radius: {diag['inside_radius_frac']:.1%} ({diag.get('inside_count', 0)}/{result['n_known']})")
            if "confidence_stats" in diag and diag["confidence_stats"].get("median") is not None:
                print(f"Median confidence: {diag['confidence_stats']['median']:.3f}")