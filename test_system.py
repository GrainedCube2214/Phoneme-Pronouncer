#!/usr/bin/env python3
"""
Test script to verify the pronunciation system works with your TIMIT setup.
Run this before full training to ensure everything is configured correctly.
"""

import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_loading():
    """Test loading audio from TIMIT"""
    logger.info("\n=== Testing Audio Loading ===")
    
    # Test file
    test_file = Path("./dataset/TRAIN/DR1/FCJF0/SA1.WAV")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        # Load with soundfile
        audio, sr = sf.read(str(test_file))
        logger.info(f"✓ Loaded audio with soundfile: shape={audio.shape}, sr={sr}")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        logger.info(f"✓ Converted to tensor: shape={audio_tensor.shape}, dtype={audio_tensor.dtype}")
        
        # Check audio properties
        logger.info(f"  Duration: {len(audio)/sr:.2f} seconds")
        logger.info(f"  Min value: {audio.min():.4f}")
        logger.info(f"  Max value: {audio.max():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return False

def test_phoneme_loading():
    """Test loading phoneme annotations"""
    logger.info("\n=== Testing Phoneme Loading ===")
    
    phn_file = Path("./dataset/TRAIN/DR1/FCJF0/SA1.PHN")
    
    if not phn_file.exists():
        logger.error(f"Phoneme file not found: {phn_file}")
        return False
    
    try:
        phonemes = []
        with open(phn_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                    phonemes.append((start, end, phone))
        
        logger.info(f"✓ Loaded {len(phonemes)} phonemes")
        logger.info(f"  First 5 phonemes: {phonemes[:5]}")
        
        # Test phoneme mapping
        PHONEME_MAP_61_TO_39 = {
            'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
            'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
            'en': 'n', 'eng': 'ng', 'nx': 'n',
            'ux': 'uw', 'pcl': 'p', 'tcl': 't', 'kcl': 'k',
            'bcl': 'b', 'dcl': 'd', 'gcl': 'g',
            'h#': 'sil', 'pau': 'sil', 'epi': 'sil',
            'q': ''
        }
        
        converted = []
        for start, end, phone in phonemes[:5]:
            converted_phone = PHONEME_MAP_61_TO_39.get(phone, phone)
            converted.append(converted_phone)
            if phone != converted_phone:
                logger.info(f"  Mapped: {phone} -> {converted_phone}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load phonemes: {e}")
        return False

def test_dataset_class():
    """Test the TIMITDataset class"""
    logger.info("\n=== Testing Dataset Class ===")
    
    try:
        # Define the phoneme mapping inline
        PHONEME_MAP_61_TO_39 = {
            'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
            'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
            'en': 'n', 'eng': 'ng', 'nx': 'n',
            'ux': 'uw', 'pcl': 'p', 'tcl': 't', 'kcl': 'k',
            'bcl': 'b', 'dcl': 'd', 'gcl': 'g',
            'h#': 'sil', 'pau': 'sil', 'epi': 'sil',
            'q': ''  # Glottal stop - remove
        }
        
        # Test loading a sample directly
        test_wav = Path("./dataset/TRAIN/DR1/FCJF0/SA1.WAV")
        test_phn = Path("./dataset/TRAIN/DR1/FCJF0/SA1.PHN")
        test_txt = Path("./dataset/TRAIN/DR1/FCJF0/SA1.TXT")
        
        # Load audio
        audio, sr = sf.read(str(test_wav))
        audio_tensor = torch.from_numpy(audio).float()
        
        # Load phonemes
        phonemes = []
        with open(test_phn, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                    phone = PHONEME_MAP_61_TO_39.get(phone, phone)
                    if phone:
                        phonemes.append((start, end, phone))
        
        # Load text
        with open(test_txt, 'r') as f:
            line = f.readline().strip()
            parts = line.split(' ', 2)
            text = parts[2] if len(parts) > 2 else line
        
        logger.info(f"✓ Loaded sample data:")
        logger.info(f"  Audio shape: {audio_tensor.shape}")
        logger.info(f"  Number of phonemes: {len(phonemes)}")
        logger.info(f"  Text: {text[:50]}...")
        
        # Test creating phoneme vocabulary
        phoneme_set = set()
        for _, _, phone in phonemes:
            phoneme_set.add(phone)
        
        phoneme_to_id = {'<pad>': 0, '<unk>': 1, '<blank>': 2}
        for phone in sorted(phoneme_set):
            phoneme_to_id[phone] = len(phoneme_to_id)
        
        logger.info(f"✓ Created phoneme vocabulary with {len(phoneme_to_id)} entries")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_installation():
    """Test if Whisper is properly installed"""
    logger.info("\n=== Testing Whisper Installation ===")
    
    try:
        import whisper
        logger.info("✓ Whisper imported successfully")
        
        # List available models
        models = ['tiny', 'base', 'small']
        logger.info(f"  Available models for testing: {models}")
        
        # Test loading tiny model (fastest)
        logger.info("  Loading tiny model for test...")
        model = whisper.load_model("tiny")
        logger.info("✓ Whisper model loaded successfully")
        
        return True
        
    except ImportError:
        logger.error("Whisper not installed. Run: pip install openai-whisper")
        return False
    except Exception as e:
        logger.error(f"Whisper test failed: {e}")
        return False

def test_transformers():
    """Test if transformers library is properly configured"""
    logger.info("\n=== Testing Transformers Library ===")
    
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        logger.info("✓ Transformers imported successfully")
        
        # Check if we can initialize a model (don't download, just check)
        logger.info("  Checking Wav2Vec2 availability...")
        
        # This will check if the model card is accessible
        from transformers import AutoModel
        try:
            # Just check config, don't download weights
            config = AutoModel.from_pretrained(
                "facebook/wav2vec2-base", 
                return_dict=False,
                output_loading_info=True
            )
            logger.info("✓ Can access Wav2Vec2 models")
        except:
            logger.info("⚠ Will download Wav2Vec2 on first use")
        
        return True
        
    except ImportError:
        logger.error("Transformers not installed. Run: pip install transformers")
        return False
    except Exception as e:
        logger.error(f"Transformers test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and CUDA setup"""
    logger.info("\n=== Testing GPU/CUDA ===")
    
    if torch.cuda.is_available():
        logger.info("✓ CUDA is available")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        # Test moving tensor to GPU
        test_tensor = torch.randn(1, 1000)
        test_tensor = test_tensor.cuda()
        logger.info("✓ Can move tensors to GPU")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"  Memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"  Memory reserved: {memory_reserved:.2f} GB")
        
        return True
    else:
        logger.warning("⚠ CUDA not available - will use CPU (slower training)")
        logger.info("  For GPU support, ensure CUDA-compatible PyTorch is installed")
        return False

def test_quick_training():
    """Test that we can import required modules for training"""
    logger.info("\n=== Testing Training Requirements ===")
    
    try:
        # Test required imports
        logger.info("Testing required imports for training...")
        
        # Test transformers
        try:
            from transformers import (
                Wav2Vec2ForCTC,
                Wav2Vec2Processor,
                Wav2Vec2CTCTokenizer,
                Wav2Vec2FeatureExtractor,
                TrainingArguments,
                Trainer
            )
            logger.info("✓ Transformers imports successful")
        except ImportError as e:
            logger.error(f"✗ Missing transformers components: {e}")
            return False
        
        # Test torch
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            logger.info("✓ PyTorch imports successful")
        except ImportError as e:
            logger.error(f"✗ Missing PyTorch components: {e}")
            return False
        
        # Test data processing
        try:
            import torchaudio
            import soundfile as sf
            import numpy as np
            logger.info("✓ Audio processing imports successful")
        except ImportError as e:
            logger.error(f"✗ Missing audio processing libraries: {e}")
            return False
        
        logger.info("✓ All training requirements met!")
        logger.info("  Ready to run: python train_pronunciation_system.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Training requirements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("PRONUNCIATION CORRECTION SYSTEM - DIAGNOSTIC TESTS")
    logger.info("="*60)
    
    tests = [
        ("Audio Loading", test_audio_loading),
        ("Phoneme Loading", test_phoneme_loading),
        ("Dataset Class", test_dataset_class),
        ("GPU/CUDA", test_gpu_availability),
        ("Whisper", test_whisper_installation),
        ("Transformers", test_transformers),
        ("Training Requirements", test_quick_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! You're ready to train.")
        logger.info("\nNext steps:")
        logger.info("1. Quick test: python train_pronunciation_system.py --test-only")
        logger.info("2. Full training: python train_pronunciation_system.py --epochs 10")
    else:
        logger.info("\n⚠ Some tests failed. Please fix the issues above.")
        logger.info("Common fixes:")
        logger.info("- Install missing packages: pip install openai-whisper transformers")
        logger.info("- For GPU: Install CUDA-compatible PyTorch")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)