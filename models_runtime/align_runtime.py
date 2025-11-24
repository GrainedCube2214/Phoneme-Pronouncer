# models_runtime/align_runtime.py
# MFA-backed alignment + small helpers. Also keeps a simple stub aligner.

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import textgrid  # pip install textgrid

# ---- MFA binary resolution ---------------------------------------------------
# Prefer explicit env var, else the hard-coded path you used, else fall back to "mfa" on PATH.
MFA_BIN = os.environ.get(
    "MFA_BIN",
    "/Users/samanthajohn/miniforge3/envs/vmfa/bin/mfa",
)
if not (Path(MFA_BIN).exists() or shutil.which(MFA_BIN)):
    # final fallback
    MFA_BIN = "mfa"

# Silence/unknown labels to skip when collecting phones
# _SIL_LABELS = {"sil", "sp", "spn", ""}
_SIL_LABELS = {"sil", ""}  # Only filter actual silence


# Very simple word tokenizer for building G2P wordlists
_WORD_RE = re.compile(r"[A-Za-z']+")

DEBUG_MFA = True


@dataclass
class PhonemeSegment:
    phoneme: str
    start_s: float
    end_s: float
    file_path: str


# -----------------------------------------------------------------------------
# Optional: tiny heuristic stub aligner (used only if you call it explicitly)
# -----------------------------------------------------------------------------
def align_phones_stub(
    audio: "np.ndarray",  # type: ignore[name-defined]
    sr: int,
    canonical_phones: List[str],
) -> List[PhonemeSegment]:
    """
    Split the utterance uniformly across the canonical phone sequence.
    Only for smoke tests; not linguistically meaningful.
    """
    dur_s = len(audio) / float(sr)
    if not canonical_phones:
        return []
    step = dur_s / len(canonical_phones)
    segs: List[PhonemeSegment] = []
    t = 0.0
    for p in canonical_phones:
        start = t
        end = min(dur_s, t + step)
        if end > start:
            segs.append(PhonemeSegment(p, start, end, file_path="(stub)"))
        t = end
    return segs


# -----------------------------------------------------------------------------
# G2P helpers: build a temporary lexicon for the sentence with MFA's G2P model
# -----------------------------------------------------------------------------
def _extract_words(text: str) -> list[str]:
    # lowercase, de-duplicate while preserving order
    seen = set()
    out = []
    for w in _WORD_RE.findall(text.lower()):
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_temp_dict_via_g2p(full_text: str) -> Optional[str]:
    """
    Use MFA's G2P to create a small pronunciation dictionary for the words
    present in `full_text`. Returns path to the dictionary or None on failure.

    Correct CLI order:
        mfa g2p INPUT_PATH G2P_MODEL_PATH OUTPUT_PATH
    """
    words = _extract_words(full_text)
    if not words:
        return None

    tmpdir = Path(tempfile.mkdtemp(prefix="mfa_g2p_"))
    words_path = tmpdir / "words.txt"
    dict_path = tmpdir / "lexicon.dict"

    # Write one word per line
    with words_path.open("w") as f:
        for w in words:
            f.write(w + "\n")

    cmd = [
        MFA_BIN,
        "g2p",
        str(words_path),        # INPUT_PATH  (file of words)
        "english_us_arpa",      # G2P model (mfa model download g2p english_us_arpa)
        str(dict_path),         # OUTPUT_PATH
        "--no_silence",
        "--include_bracketed", "false",
        "--include_punctuation", "false",
    ]
    print(f"[MFA] G2P cmd: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("[MFA] G2P failed. stderr:\n", e.stderr.decode("utf-8", errors="ignore"))
        return None

    if not dict_path.exists():
        print("[MFA] G2P produced no dictionary file.")
        return None

    return str(dict_path)


# -----------------------------------------------------------------------------
# Main MFA aligner
# -----------------------------------------------------------------------------
def align_phones_mfa(
    wav_path: str,
    transcript: str,
    target_word: Optional[str] = None,
    prefer_sidecar: bool = True,  # ADD THIS PARAMETER
) -> List[PhonemeSegment]:
    """
    Align `wav_path` to `transcript` with MFA and return list of phone segments.
    """
    # If a TIMIT-style PHN file sits next to the WAV, use it (ground truth)
    phn_sidecar = os.path.splitext(wav_path)[0] + ".PHN"
    if os.path.exists(phn_sidecar) and prefer_sidecar:  # Respect prefer_sidecar flag
        segs = []
        with open(phn_sidecar) as f:
            for line in f:
                a, b, lab = line.strip().split()
                if lab.lower() in _SIL_LABELS:
                    continue
                segs.append(PhonemeSegment(
                    phoneme=lab.lower(),
                    start_s=float(a) / 16000.0,
                    end_s=float(b) / 16000.0,
                    file_path=wav_path,
                ))
        if segs:
            print(f"[MFA] Using sidecar PHN alignments: {len(segs)} phones")
        return segs

    # Debug: print what we're trying to align
    print(f"[MFA DEBUG] Aligning transcript: '{transcript}'")
    print(f"[MFA DEBUG] Normalized: '{_norm_text(transcript)}'")

    # Create tiny corpus for a single file
    with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as corpus_dir, \
         tempfile.TemporaryDirectory(prefix="mfa_out_") as out_dir:

        corpus_dir = Path(corpus_dir)
        out_dir = Path(out_dir)
        base = Path(wav_path).stem

        wav_copy = corpus_dir / f"{base}.wav"
        lab_path = corpus_dir / f"{base}.lab"

        # Copy audio
        shutil.copy2(wav_path, wav_copy)

        # Write transcript
        lab_path.write_text(_norm_text(transcript) + "\n")

        # Try G2P first
        dict_path = _build_temp_dict_via_g2p(transcript)
        dictionary_arg = dict_path if dict_path is not None else "english_us_mfa"

        # MFA align command
        cmd = [
            MFA_BIN,
            "align",
            str(corpus_dir),
            dictionary_arg,
            "english_mfa",
            str(out_dir),
            "--clean",
            "--beam", "400",           # ADD: Wider search beam
            "--retry_beam", "800", 
        ]
        print(f"[MFA] Running: {' '.join(cmd)}", flush=True)

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[MFA] Alignment failed: {e}")
            if e.stderr:
                print("[MFA] stderr:\n", e.stderr.decode("utf-8", errors="ignore"))
            return []

        tg_path = out_dir / f"{base}.TextGrid"
        if not tg_path.exists():
            print(f"[MFA] No TextGrid found at {tg_path}")
            return []

        # Parse TextGrid
        tg = textgrid.TextGrid.fromFile(str(tg_path))

        # DEBUG: Show what tiers exist
        print(f"[MFA DEBUG] TextGrid has {len(tg.tiers)} tiers")
        for tier in tg.tiers:
            print(f"[MFA DEBUG] Tier: '{tier.name}' with {len(tier.intervals)} intervals")

        # Find tiers (case-insensitive)
        word_tier = None
        phone_tier = None
        for tier in tg.tiers:
            name = (tier.name or "").strip().lower()
            if name == "words":
                word_tier = tier
            elif name == "phones":
                phone_tier = tier

        if phone_tier is None:
            print("[MFA] No 'phones' tier found in TextGrid.")
            return []

        # DEBUG: Show phone intervals
        print(f"[MFA DEBUG] Phone tier intervals (first 20):")
        for itv in phone_tier.intervals[:20]:
            ph = (itv.mark or "").strip().lower()
            print(f"  {itv.minTime:.3f}-{itv.maxTime:.3f}: '{ph}' (in SIL_LABELS: {ph in _SIL_LABELS})")

        # Handle target word clipping
        word_start = None
        word_end = None
        if target_word and word_tier:
            tw = target_word.lower()
            for itv in word_tier.intervals:
                if (itv.mark or "").strip().lower() == tw:
                    word_start = float(itv.minTime)
                    word_end = float(itv.maxTime)
                    break
            if word_start is None:
                print(f"[MFA] Target word '{target_word}' not found in word tier; will use full utterance phones.")

        # Gather non-silence phones
        segs: List[PhonemeSegment] = []
        for itv in phone_tier.intervals:
            ph = (itv.mark or "").strip().lower()
            if ph in _SIL_LABELS:
                continue

            start_s = float(itv.minTime)
            end_s = float(itv.maxTime)
            if end_s <= start_s:
                continue

            if word_start is not None and word_end is not None:
                if end_s <= word_start or start_s >= word_end:
                    continue

            segs.append(PhonemeSegment(phoneme=ph, start_s=start_s, end_s=end_s, file_path=wav_path))

        # Log result
        if segs:
            print(f"[MFA] Returning {len(segs)} phone segments")
        else:
            print("[MFA] Alignment succeeded but yielded 0 non-silence phones in the chosen span.")
        return segs