#!/usr/bin/env python3
"""
Complete training pipeline for TIMIT-based pronunciation correction system.
Save this as: train_pronunciation_system.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

@dataclass
class TIMITSample:
    """Single TIMIT utterance with all annotations"""
    audio_path: str
    wav: torch.Tensor
    sample_rate: int
    text: str
    words: List[tuple]
    phonemes: List[tuple]
    speaker_id: str
    dialect: str
    gender: str

class TIMITDataset(Dataset):
    """TIMIT Dataset for PyTorch"""
    
    PHONEME_MAP_61_TO_39 = {
        'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
        'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
        'en': 'n', 'eng': 'ng', 'nx': 'n',
        'ux': 'uw', 'pcl': 'p', 'tcl': 't', 'kcl': 'k',
        'bcl': 'b', 'dcl': 'd', 'gcl': 'g',
        'h#': 'sil', 'pau': 'sil', 'epi': 'sil',
        'q': ''  # Glottal stop - remove
    }
    
    def __init__(self, timit_root: str, split: str = 'TRAIN', use_39_phonemes: bool = True):
        self.timit_root = Path(timit_root)
        self.split = split.upper()
        self.use_39_phonemes = use_39_phonemes
        self.samples = []
        self.phoneme_to_id = {}
        
        # Load samples
        self._load_samples()
        
        # Build phoneme vocabulary
        self._build_phoneme_vocab()
        
        logger.info(f"Loaded {len(self.samples)} samples from {split}")
        logger.info(f"Phoneme vocabulary size: {len(self.phoneme_to_id)}")
    
    def _load_samples(self):
        """Load all samples from the dataset"""
        split_dir = self.timit_root / self.split
        
        for wav_path in split_dir.glob('**/*.WAV'):
            try:
                sample = self._load_utterance(wav_path)
                self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {wav_path}: {e}")
    
    def _load_utterance(self, wav_path: Path) -> TIMITSample:
        """Load a single utterance"""
        # Parse metadata from path
        parts = wav_path.parts
        dialect = parts[-3]
        speaker_dir = parts[-2]
        gender = speaker_dir[0].lower()
        speaker_id = speaker_dir[1:]
        
        # Load audio using soundfile (handles SPHERE format)
        audio, sr = sf.read(str(wav_path))
        audio = torch.from_numpy(audio).float()
        
        # Ensure mono
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        
        # Load text
        txt_path = wav_path.with_suffix('.TXT')
        with open(txt_path, 'r') as f:
            line = f.readline().strip()
            parts = line.split(' ', 2)
            text = parts[2] if len(parts) > 2 else line
        
        # Load phonemes
        phonemes = []
        phn_path = wav_path.with_suffix('.PHN')
        with open(phn_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                    if self.use_39_phonemes:
                        phone = self.PHONEME_MAP_61_TO_39.get(phone, phone)
                    if phone:  # Skip empty
                        phonemes.append((start, end, phone))
        
        # Load words
        words = []
        wrd_path = wav_path.with_suffix('.WRD')
        with open(wrd_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start, end, word = int(parts[0]), int(parts[1]), parts[2]
                    words.append((start, end, word))
        
        return TIMITSample(
            audio_path=str(wav_path),
            wav=audio,
            sample_rate=sr,
            text=text,
            words=words,
            phonemes=phonemes,
            speaker_id=speaker_id,
            dialect=dialect,
            gender=gender
        )
    
    def _build_phoneme_vocab(self):
        """Build phoneme to ID mapping"""
        phonemes = set()
        for sample in self.samples:
            for _, _, phone in sample.phonemes:
                phonemes.add(phone)
        
        # Special tokens
        self.phoneme_to_id = {
            '<pad>': 0,
            '<unk>': 1,
            '<blank>': 2  # For CTC
        }
        
        # Add phonemes
        for phone in sorted(phonemes):
            if phone not in self.phoneme_to_id:
                self.phoneme_to_id[phone] = len(self.phoneme_to_id)
        
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert phonemes to IDs
        phoneme_ids = [self.phoneme_to_id.get(p[2], self.phoneme_to_id['<unk>']) 
                      for p in sample.phonemes]
        
        return {
            'audio': sample.wav,
            'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
            'text': sample.text,
            'sample_rate': sample.sample_rate
        }

@dataclass
class DataCollatorForCTC:
    """Data collator for CTC training"""
    
    processor: Optional[Wav2Vec2Processor] = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict]) -> Dict:
        # Extract audio and labels
        audio_features = []
        labels = []
        
        for feature in features:
            # Process audio to 16kHz if needed
            audio = feature['audio']
            if feature['sample_rate'] != 16000:
                # Resample
                audio = torchaudio.functional.resample(
                    audio, feature['sample_rate'], 16000
                )
            
            audio_features.append({'input_values': audio})
            labels.append(feature['phoneme_ids'])
        
        # Pad audio
        batch = self.processor.pad(
            audio_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Pad labels
        label_features = [{"input_ids": label} for label in labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch

# ============================================================================
# Model Training
# ============================================================================

class Wav2Vec2PhonemeTrainer:
    """Train Wav2Vec2 for phoneme recognition on TIMIT"""
    
    def __init__(self, 
                 train_dataset: TIMITDataset,
                 eval_dataset: TIMITDataset,
                 output_dir: str = "./models/wav2vec2-timit",
                 model_name: str = "facebook/wav2vec2-base"):
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True
        )
        
        # Create tokenizer from phoneme vocabulary
        vocab_dict = train_dataset.id_to_phoneme
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_dict=vocab_dict,
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|"
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer
        )
        
        # Initialize model
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(train_dataset.phoneme_to_id)
        )
        
        # Freeze feature extractor initially
        self.model.freeze_feature_encoder()
        
        logger.info(f"Model initialized with vocab size: {len(train_dataset.phoneme_to_id)}")
    
    def compute_metrics(self, eval_pred):
        """Compute Phoneme Error Rate"""
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        label_ids = eval_pred.label_ids
        
        # Decode predictions
        decoded_preds = []
        decoded_labels = []
        
        for pred, label in zip(predictions, label_ids):
            # Remove special tokens
            pred_str = self.processor.batch_decode(pred)
            label_str = self.processor.batch_decode(label[label != -100])
            
            decoded_preds.append(pred_str)
            decoded_labels.append(label_str)
        
        # Calculate PER (simplified)
        total_errors = 0
        total_phonemes = 0
        
        for pred, ref in zip(decoded_preds, decoded_labels):
            # Simple error counting
            errors = sum(1 for p, r in zip(pred, ref) if p != r)
            total_errors += errors
            total_phonemes += len(ref)
        
        per = total_errors / max(total_phonemes, 1)
        
        return {"per": per}
    
    def train(self, num_epochs: int = 10, batch_size: int = 8):
        """Train the model"""
        
        # Data collator
        data_collator = DataCollatorForCTC(processor=self.processor)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            warmup_steps=500,
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none"  # Disable wandb/tensorboard for now
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(str(self.output_dir))
        
        # Save phoneme vocabulary
        vocab_path = self.output_dir / "phoneme_vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(self.train_dataset.phoneme_to_id, f, indent=2)
        
        logger.info(f"Model saved to {self.output_dir}")
        
        return trainer

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train pronunciation correction system on TIMIT")
    parser.add_argument('--timit-root', type=str, default='./dataset',
                       help='Path to TIMIT dataset root')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size (reduce if OOM)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test loading data, don\'t train')
    
    args = parser.parse_args()
    
    # Verify TIMIT path
    timit_path = Path(args.timit_root)
    if not timit_path.exists():
        logger.error(f"TIMIT path not found: {timit_path}")
        sys.exit(1)
    
    logger.info(f"Using TIMIT dataset at: {timit_path}")
    
    # Load datasets
    logger.info("Loading training data...")
    train_dataset = TIMITDataset(args.timit_root, split='TRAIN', use_39_phonemes=True)
    
    logger.info("Loading test data...")
    test_dataset = TIMITDataset(args.timit_root, split='TEST', use_39_phonemes=True)
    
    if args.test_only:
        # Test data loading
        logger.info("\n=== Data Loading Test ===")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Test one sample
        sample = train_dataset[0]
        logger.info(f"Sample audio shape: {sample['audio'].shape}")
        logger.info(f"Sample phonemes: {sample['phoneme_ids'].shape}")
        logger.info(f"Sample text: {sample['text']}")
        
        logger.info("\n✅ Data loading successful!")
        return
    
    # Train model
    logger.info("\n=== Starting Training ===")
    trainer = Wav2Vec2PhonemeTrainer(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        output_dir=f"{args.output_dir}/wav2vec2-timit"
    )
    
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Model saved to: {args.output_dir}/wav2vec2-timit")
    
    # Save configuration
    config = {
        'timit_root': str(timit_path),
        'model_dir': f"{args.output_dir}/wav2vec2-timit",
        'phoneme_vocab': f"{args.output_dir}/wav2vec2-timit/phoneme_vocab.json",
        'num_phonemes': len(train_dataset.phoneme_to_id),
        'training_samples': len(train_dataset),
        'test_samples': len(test_dataset)
    }
    
    config_path = Path(args.output_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    main()