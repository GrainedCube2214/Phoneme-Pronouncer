# Zero-Shot Phoneme-Level Pronunciation Checker

_Outdated, see report.pdf instead for setup instructions_

A real-time pronunciation feedback system using pre-trained Wav2Vec2-Large embeddings and prototype-based classification. Achieves **94.1% accuracy** on TIMIT benchmark with zero task-specific training.

---

## Quick Start

### Requirements
- **Python 3.10** (required)
- Microphone for live recording

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('cmudict')"
```

### Run

```bash
python live_checker_nomfa.py
```

Press Enter, speak naturally, and get instant phoneme-level pronunciation feedback!

---

## Sample Output

```
🎤 Recording for 5 seconds... SPEAK NOW!
✓ Recording complete!

======================================================================
ZERO-SHOT PRONUNCIATION ANALYSIS
======================================================================

You said: "she had your dark suit in greasy washwater all year"

🌟 Clarity Score: 92% (24/26 phonemes clear)
Excellent clarity!

Detected Phonemes:
  sh (0.85) iy(0.92) ae(0.88) er(0.91) d (0.79) aa(0.94) r (0.87) ...

Unclear phonemes (practice these):
  'k' at 1.4s (confidence: 0.18)
  'l' at 3.4s (confidence: 0.21)

======================================================================
```

---

## Performance

**TIMIT Test Set (9,641 phonemes, 500 files):**

| Metric | Accuracy |
|--------|----------|
| **Overall** | **94.1%** |
| Fricatives (s, sh, f, v, z) | 96.0% |
| Vowels (all) | 94.7% |
| Nasals (m, n, ng) | 94.6% |
| Liquids (l, r, er) | 94.1% |
| Stops (p, t, k, b, d, g) | 89.2% |
| Affricates (ch, jh) | 96.6% |

**Key Metrics:**
- Distance separation: Correct (0.41) vs Incorrect (0.71)
- Confidence separation: Correct (0.41) vs Incorrect (0.21)
- Inside P95 radius: 96.2% calibration accuracy

---

## Architecture

```
Audio Input (16kHz)
    ↓
ASR (Wav2Vec2-CTC) → Detect what was said
    ↓
Sliding Window Segmentation → 120ms windows with 40ms hop
    ↓
Wav2Vec2-Large Encoder → 1024D L2-normalized embeddings
    ↓
Prototype Classifier → Compare to 39 phoneme prototypes
    ↓
Confidence-based Scoring → Per-phoneme feedback
```

---

## Project Structure

```
├── app.py                           # Core evaluation pipeline
├── live_checker_nomfa.py           # Live pronunciation practice (START HERE)
├── batch_evaluate_timit_memory_safe.py  # Batch evaluation
├── diagnostic_tools.py             # Confusion matrix, visualizations
├── requirements.txt                # Python dependencies
│
├── models_runtime/
│   ├── asr_runtime.py             # Wav2Vec2-CTC for transcription
│   ├── embedding_runtime.py       # Wav2Vec2-Large embedder (L2-normalized)
│   ├── phonemespace_runtime.py    # Prototype-based classifier
│   ├── align_runtime.py           # MFA wrapper (for TIMIT evaluation)
│   └── cmu_lexicon.py             # CMU pronunciation dictionary
│
└── artifacts/
    ├── phoneme_prototypes_large.pkl   # Pre-trained prototypes (39 phonemes, 120K segments)
    └── phoneme_prototypes_large.json  # Statistics (P95 radii: 0.58-0.73)
```

---

## Usage

### Live Pronunciation Practice

```bash
python live_checker_nomfa.py
```

- Record from microphone
- Get instant phoneme-level feedback
- See which sounds need practice
- Zero-shot: works with any speech

### Batch Evaluation (TIMIT)

```bash
# Evaluate 100 test files
python batch_evaluate_timit_memory_safe.py data_new/TEST --max-files 100

# Outputs: timit_test_results_large.json
```

### Generate Diagnostics

```bash
# After running app.py to create result.json:
python diagnostic_tools.py result.json \
    --proto-pkl artifacts/phoneme_prototypes_large.pkl \
    --output-dir diagnostics_output

# Generates:
# - confusion_matrix.png
# - distance_analysis.png
# - per_phoneme_accuracy.png
# - summary_report.txt
```

---

## Technical Details

### Decision Rule

For each phoneme segment:

1. **Embedding**: Extract 1024D Wav2Vec2-Large feature, L2-normalize to unit sphere
2. **Distance**: Compute L2 distance to target phoneme prototype centroid
3. **Radius check**: Accept if `distance ≤ p95_radius` (95th percentile from 120K training segments)
4. **Confidence threshold**: 
   - 0.20 for cross-family (e.g., vowel vs consonant)
   - 0.12 for same-family (e.g., vowel vs vowel)
5. **Correctness**: `(inside_radius AND confidence ≥ threshold)`

### Phoneme Set (39 phones)

```
Vowels:    iy ih eh ey ae aa ah uh uw ow oy aw ay er
Stops:     p b t d k g
Fricatives: f v th dh s z sh hh
Affricates: ch jh
Nasals:    m n ng
Liquids:   l r
Glides:    w y
Flap:      dx
```

### Normalization Formula

```python
embedding_normalized = embedding / ||embedding||_2
```

All vectors projected to unit hypersphere (norm = 1.0), making distances measure angular similarity rather than absolute position.

---


## Training (Optional)

Pre-trained prototypes are included. To rebuild from scratch:

1. Obtain TIMIT corpus (~4,620 training files)
2. Run `rebuild_prototypes_large.ipynb` (3-4 hours)
3. Generates new prototypes in `artifacts/`

**Not required for inference.**

---

## Citation

```bibtex
@misc{pronunciation-checker-2025,
  title={Zero-Shot Phoneme-Level Pronunciation Checker},
  author={Your Name},
  year={2025},
  note={94.1\% accuracy on TIMIT using Wav2Vec2-Large}
}
```

**Based on:**
- Wav2Vec 2.0 (Baevski et al., 2020)
- Montreal Forced Aligner (McAuliffe et al., 2017)
- TIMIT Corpus (Garofolo et al., 1993)

---

## Acknowledgments

- Pre-trained Wav2Vec2 models from Meta AI
- Montreal Forced Aligner from Montreal Corpus Tools
- TIMIT acoustic-phonetic continuous speech corpus