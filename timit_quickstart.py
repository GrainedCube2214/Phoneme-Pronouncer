#!/usr/bin/env python3
"""
Quick start script for TIMIT pronunciation correction system.
Run this first to verify your TIMIT dataset is working correctly.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_timit_structure(timit_root: str) -> bool:
    """Verify TIMIT dataset structure"""
    timit_path = Path(timit_root)
    
    required_dirs = ['TRAIN', 'TEST', 'DOC']
    for dir_name in required_dirs:
        if not (timit_path / dir_name).exists():
            logger.error(f"Missing required directory: {dir_name}")
            return False
    
    # Check for some sample files
    sample_file = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'SA1.WAV'
    if not sample_file.exists():
        # Try lowercase
        sample_file = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'sa1.wav'
    
    if not sample_file.exists():
        logger.error(f"Cannot find sample file: {sample_file}")
        logger.info("Please check if your files use .WAV or .wav extension")
        return False
    
    logger.info("✓ TIMIT structure verified")
    return True

def quick_load_sample(timit_root: str, uppercase: bool = True) -> Dict:
    """Load a single TIMIT sample for testing"""
    timit_path = Path(timit_root)
    
    # Try to load SA1 from first speaker
    if uppercase:
        wav_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'SA1.WAV'
        txt_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'SA1.TXT'
        phn_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'SA1.PHN'
        wrd_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'SA1.WRD'
    else:
        wav_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'sa1.wav'
        txt_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'sa1.txt'
        phn_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'sa1.phn'
        wrd_path = timit_path / 'TRAIN' / 'DR1' / 'FCJF0' / 'sa1.wrd'
    
    if not wav_path.exists():
        # Try opposite case
        return quick_load_sample(timit_root, not uppercase)
    
    # Load audio
    # TIMIT uses NIST SPHERE format, we need to handle this
    try:
        # Try with torchaudio first
        wav, sr = torchaudio.load(str(wav_path))
        logger.info(f"Loaded audio: shape={wav.shape}, sr={sr}")
    except:
        # If torchaudio fails, try with soundfile or sox
        logger.warning("torchaudio failed, trying alternative loader...")
        import soundfile as sf
        wav, sr = sf.read(str(wav_path))
        wav = torch.from_numpy(wav).float().unsqueeze(0)
    
    # Load text
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split(' ', 2)
        text = parts[2] if len(parts) > 2 else line
    
    # Load phonemes
    phonemes = []
    with open(phn_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                phonemes.append((start, end, phone))
    
    # Load words
    words = []
    with open(wrd_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start, end, word = int(parts[0]), int(parts[1]), parts[2]
                words.append((start, end, word))
    
    return {
        'audio': wav,
        'sample_rate': sr,
        'text': text,
        'phonemes': phonemes,
        'words': words,
        'path': str(wav_path)
    }

def analyze_timit_stats(timit_root: str) -> Dict:
    """Analyze TIMIT dataset statistics"""
    timit_path = Path(timit_root)
    
    stats = {
        'train_speakers': 0,
        'test_speakers': 0,
        'train_utterances': 0,
        'test_utterances': 0,
        'dialects': {},
        'phoneme_set': set(),
        'sentence_types': {'SA': 0, 'SI': 0, 'SX': 0}
    }
    
    # Count training data
    train_dir = timit_path / 'TRAIN'
    for dialect_dir in train_dir.iterdir():
        if dialect_dir.is_dir() and dialect_dir.name.startswith('DR'):
            dialect = dialect_dir.name
            if dialect not in stats['dialects']:
                stats['dialects'][dialect] = {'train': 0, 'test': 0}
            
            for speaker_dir in dialect_dir.iterdir():
                if speaker_dir.is_dir():
                    stats['train_speakers'] += 1
                    stats['dialects'][dialect]['train'] += 1
                    
                    # Count utterances
                    wav_files = list(speaker_dir.glob('*.WAV'))
                    if not wav_files:
                        wav_files = list(speaker_dir.glob('*.wav'))
                    stats['train_utterances'] += len(wav_files)
                    
                    # Count sentence types
                    for wav_file in wav_files:
                        sent_type = wav_file.stem[:2].upper()
                        if sent_type in stats['sentence_types']:
                            stats['sentence_types'][sent_type] += 1
    
    # Count test data
    test_dir = timit_path / 'TEST'
    for dialect_dir in test_dir.iterdir():
        if dialect_dir.is_dir() and dialect_dir.name.startswith('DR'):
            dialect = dialect_dir.name
            if dialect not in stats['dialects']:
                stats['dialects'][dialect] = {'train': 0, 'test': 0}
            
            for speaker_dir in dialect_dir.iterdir():
                if speaker_dir.is_dir():
                    stats['test_speakers'] += 1
                    stats['dialects'][dialect]['test'] += 1
                    
                    wav_files = list(speaker_dir.glob('*.WAV'))
                    if not wav_files:
                        wav_files = list(speaker_dir.glob('*.wav'))
                    stats['test_utterances'] += len(wav_files)
    
    return stats

def create_phoneme_mapping() -> Dict[str, str]:
    """Create the standard 61-to-39 phoneme mapping for TIMIT"""
    
    # Standard TIMIT 61-to-39 phoneme mapping
    mapping = {
        # Vowels
        'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ey': 'ey',
        'ae': 'ae', 'aa': 'aa', 'aw': 'aw', 'ay': 'ay',
        'ah': 'ah', 'ao': 'aa', 'oy': 'oy', 'ow': 'ow',
        'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'er': 'er',
        'ax': 'ah', 'ix': 'ih', 'axr': 'er', 'ax-h': 'ah',
        
        # Stops
        'b': 'b', 'bcl': 'b', 'd': 'd', 'dcl': 'd',
        'g': 'g', 'gcl': 'g', 'p': 'p', 'pcl': 'p',
        't': 't', 'tcl': 't', 'k': 'k', 'kcl': 'k',
        'dx': 'dx', 'q': 'q',
        
        # Affricates
        'jh': 'jh', 'ch': 'ch',
        
        # Fricatives
        's': 's', 'sh': 'sh', 'z': 'z', 'zh': 'zh',
        'f': 'f', 'th': 'th', 'v': 'v', 'dh': 'dh',
        
        # Nasals
        'm': 'm', 'em': 'm', 'n': 'n', 'en': 'n',
        'ng': 'ng', 'eng': 'ng', 'nx': 'n',
        
        # Liquids and Glides
        'l': 'l', 'el': 'l', 'r': 'r', 'w': 'w',
        'y': 'y', 'hh': 'hh', 'hv': 'hh',
        
        # Others
        'h#': 'sil', 'pau': 'sil', 'epi': 'sil',
        '#h': 'sil', '!ENTER': 'sil', '!EXIT': 'sil'
    }
    
    return mapping

def test_phoneme_conversion(sample: Dict) -> None:
    """Test phoneme conversion on a sample"""
    
    mapping = create_phoneme_mapping()
    
    logger.info("\n=== Phoneme Conversion Test ===")
    logger.info(f"Original text: {sample['text']}")
    logger.info(f"Number of phonemes: {len(sample['phonemes'])}")
    
    # Convert phonemes
    original_phones = [p[2] for p in sample['phonemes']]
    converted_phones = [mapping.get(p, p) for p in original_phones]
    
    # Show first 10 phonemes
    logger.info("\nFirst 10 phonemes (61 set -> 39 set):")
    for i in range(min(10, len(original_phones))):
        orig = original_phones[i]
        conv = converted_phones[i]
        if orig != conv:
            logger.info(f"  {orig:6s} -> {conv:6s} (converted)")
        else:
            logger.info(f"  {orig:6s} -> {conv:6s}")
    
    # Count unique phonemes
    unique_61 = len(set(original_phones))
    unique_39 = len(set(converted_phones))
    logger.info(f"\nUnique phonemes: {unique_61} (61-set) -> {unique_39} (39-set)")

def save_configuration(timit_root: str, output_dir: str = "./config") -> None:
    """Save configuration files for the system"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save paths configuration
    config = {
        'timit_root': str(Path(timit_root).absolute()),
        'train_dir': str(Path(timit_root).absolute() / 'TRAIN'),
        'test_dir': str(Path(timit_root).absolute() / 'TEST'),
        'doc_dir': str(Path(timit_root).absolute() / 'DOC'),
        'uppercase_extensions': True  # Set based on your dataset
    }
    
    config_file = output_path / 'timit_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_file}")
    
    # Save phoneme mapping
    mapping = create_phoneme_mapping()
    mapping_file = output_path / 'phoneme_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"Phoneme mapping saved to {mapping_file}")

def main():
    """Main function to test TIMIT setup"""
    
    # Get TIMIT root directory
    if len(sys.argv) > 1:
        timit_root = sys.argv[1]
    else:
        # Try to find TIMIT in common locations
        possible_paths = [
            './dataset',
            './TIMIT',
            './timit',
            '../TIMIT',
            '../timit',
            '~/TIMIT',
            '~/timit'
        ]
        
        timit_root = None
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                timit_root = expanded_path
                break
        
        if not timit_root:
            print("Usage: python timit_quickstart.py /path/to/TIMIT")
            print("\nCouldn't find TIMIT dataset. Please provide the path.")
            sys.exit(1)
    
    logger.info(f"Using TIMIT root: {timit_root}")
    
    # Step 1: Verify structure
    if not verify_timit_structure(timit_root):
        sys.exit(1)
    
    # Step 2: Load a sample
    logger.info("\n=== Loading Sample ===")
    sample = quick_load_sample(timit_root)
    logger.info(f"✓ Successfully loaded sample from {sample['path']}")
    
    # Step 3: Analyze dataset
    logger.info("\n=== Dataset Statistics ===")
    stats = analyze_timit_stats(timit_root)
    logger.info(f"Training speakers: {stats['train_speakers']}")
    logger.info(f"Test speakers: {stats['test_speakers']}")
    logger.info(f"Training utterances: {stats['train_utterances']}")
    logger.info(f"Test utterances: {stats['test_utterances']}")
    logger.info(f"Dialects: {len(stats['dialects'])}")
    logger.info(f"Sentence types: SA={stats['sentence_types']['SA']}, "
                f"SI={stats['sentence_types']['SI']}, "
                f"SX={stats['sentence_types']['SX']}")
    
    # Step 4: Test phoneme conversion
    test_phoneme_conversion(sample)
    
    # Step 5: Save configuration
    logger.info("\n=== Saving Configuration ===")
    save_configuration(timit_root)
    
    logger.info("\n✅ TIMIT setup verification complete!")
    logger.info("You can now proceed with training the pronunciation correction system.")
    
    # Optional: Test audio playback
    try:
        import IPython.display as ipd
        logger.info("\n=== Audio Test ===")
        logger.info(f"Sample text: '{sample['text']}'")
        logger.info("If in Jupyter, audio would play here")
        # ipd.Audio(sample['audio'].numpy(), rate=sample['sample_rate'])
    except ImportError:
        pass
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)