# models_runtime/asr_runtime.py

from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAMPLE_RATE = 16000

# Choose device once
if torch.backends.mps.is_available():
    _DEVICE = "mps"
elif torch.cuda.is_available():
    _DEVICE = "cuda"
else:
    _DEVICE = "cpu"

print(f"[ASR] Using device: {_DEVICE}")

_ASR_PROCESSOR = None
_ASR_MODEL = None


def load_wav_16k(path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as 16kHz mono float32 in [-1, 1].
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    # make mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # resample if needed
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr)

    return audio.astype("float32"), SAMPLE_RATE


def _get_asr():
    """
    Lazy-load Wav2Vec2 processor + CTC model without using HF pipelines.
    This avoids torchcodec and the register_fake issue.
    """
    global _ASR_PROCESSOR, _ASR_MODEL
    if _ASR_PROCESSOR is None:
        model_name = "facebook/wav2vec2-base-960h"
        _ASR_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
        _ASR_MODEL = Wav2Vec2ForCTC.from_pretrained(model_name).to(_DEVICE)
        _ASR_MODEL.eval()
    return _ASR_PROCESSOR, _ASR_MODEL


def transcribe_word(path: str) -> str:
    """
    Very simple English ASR using Wav2Vec2 CTC.
    Returns raw text string (lowercased).
    """
    audio, sr = load_wav_16k(path)
    processor, model = _get_asr()

    with torch.no_grad():
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        logits = model(inputs.input_values.to(_DEVICE)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return text.strip().lower()
