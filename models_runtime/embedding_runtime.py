# models_runtime/embedding_runtime.py
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

SAMPLE_RATE = 16000

class SegmentEmbedder:
    def __init__(self, model_name="facebook/wav2vec2-large-960h", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use safetensors to avoid PyTorch version conflict
        self.model = Wav2Vec2Model.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(self.device)
        
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed_segments(self, seg_waves: list[np.ndarray]) -> np.ndarray:
        """
        seg_waves: list of 1D float32 numpy arrays (16kHz)
        Returns: (B, D) L2-normalized embeddings.
        """
        inputs = self.proc(
            seg_waves,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        out = self.model(
            inputs.input_values.to(self.device),
            output_hidden_states=True
        )
        
        # Mean-pool over time: (B, T, D) -> (B, D)
        embs = out.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
        
        # CRITICAL: L2-normalize to unit sphere
        # This makes distances stable and comparable
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_normalized = embs / (norms + 1e-8)
        
        return embs_normalized