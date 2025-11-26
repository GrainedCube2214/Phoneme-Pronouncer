#!/usr/bin/env python3
"""
Branch 2: STT + Semantic Validation
Uses Whisper for transcription and BERT/Sentence-BERT for lightweight validation
No LLM needed - fast and efficient!
"""

import numpy as np
import whisper
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import Levenshtein
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass

@dataclass
class SemanticValidationResult:
    """Result from semantic validation"""
    transcription: str
    is_semantically_valid: bool
    similarity_score: float
    word_error_rate: float
    is_informal_variant: bool
    substitutions: List[Tuple[str, str]]
    confidence: float
    feedback: str

class SemanticValidator:
    """
    Lightweight semantic validation using STT + BERT
    No LLM required - runs locally and fast!
    """
    
    def __init__(self, 
                 stt_model: str = "base",
                 similarity_model: str = "all-MiniLM-L6-v2",
                 device: str = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Whisper for STT
        print(f"Loading Whisper model: {stt_model}")
        self.whisper = whisper.load_model(stt_model, device=self.device)
        
        # Load Sentence-BERT for semantic similarity
        print(f"Loading similarity model: {similarity_model}")
        self.similarity_model = SentenceTransformer(similarity_model)
        
        # Common informal variations dictionary
        self.informal_variations = {
            "going to": ["gonna", "goin' to", "goin to", "gon'"],
            "want to": ["wanna", "wanta", "wan'"],
            "got to": ["gotta", "got ta"],
            "have to": ["hafta", "hasta"],
            "kind of": ["kinda", "kind'a"],
            "sort of": ["sorta", "sort'a"],
            "out of": ["outta", "out'a"],
            "a lot of": ["alotta", "a lotta"],
            "don't know": ["dunno"],
            "let me": ["lemme"],
            "give me": ["gimme"],
            "about to": ["boutta", "'bout to"],
            "trying to": ["tryna", "tryin' to"],
            "supposed to": ["s'posed to", "supposta"],
            "because": ["'cause", "cuz", "cos"],
            "probably": ["prolly", "prob'ly"],
            "actually": ["act'lly"],
            "comfortable": ["comfy"],
            "until": ["'til", "till"],
        }
        
        # Filler words to ignore in validation
        self.filler_words = {
            "uh", "um", "ah", "er", "like", "you know", 
            "well", "so", "basically", "literally"
        }
        
    def validate(self, 
                 audio: np.ndarray,
                 target_text: str,
                 sample_rate: int = 16000) -> SemanticValidationResult:
        """
        Main validation pipeline
        """
        
        # Step 1: Transcribe audio
        transcription = self._transcribe(audio, sample_rate)
        
        # Step 2: Clean and normalize texts
        clean_trans = self._clean_text(transcription)
        clean_target = self._clean_text(target_text)
        
        # Step 3: Check for exact match
        if clean_trans.lower() == clean_target.lower():
            return SemanticValidationResult(
                transcription=transcription,
                is_semantically_valid=True,
                similarity_score=1.0,
                word_error_rate=0.0,
                is_informal_variant=False,
                substitutions=[],
                confidence=1.0,
                feedback="Perfect match!"
            )
        
        # Step 4: Check for informal variations
        norm_trans = self._normalize_informal(clean_trans)
        norm_target = self._normalize_informal(clean_target)
        
        is_informal = (norm_trans.lower() == norm_target.lower() and 
                      clean_trans.lower() != clean_target.lower())
        
        # Step 5: Calculate similarity metrics
        similarity = self._calculate_similarity(transcription, target_text)
        wer = self._calculate_wer(clean_trans, clean_target)
        
        # Step 6: Find word substitutions
        substitutions = self._find_substitutions(clean_trans, clean_target)
        
        # Step 7: Determine validity
        is_valid = self._determine_validity(similarity, wer, is_informal)
        
        # Step 8: Calculate confidence
        confidence = self._calculate_confidence(similarity, wer, is_informal)
        
        # Step 9: Generate feedback
        feedback = self._generate_feedback(
            is_valid, similarity, wer, is_informal, substitutions
        )
        
        return SemanticValidationResult(
            transcription=transcription,
            is_semantically_valid=is_valid,
            similarity_score=similarity,
            word_error_rate=wer,
            is_informal_variant=is_informal,
            substitutions=substitutions,
            confidence=confidence,
            feedback=feedback
        )
    
    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio using Whisper"""
        
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        # Transcribe
        result = self.whisper.transcribe(
            audio,
            language="en",
            fp16=(self.device == "cuda")
        )
        
        return result["text"].strip()
    
    def _clean_text(self, text: str) -> str:
        """Remove punctuation and extra spaces"""
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", "", text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text
    
    def _normalize_informal(self, text: str) -> str:
        """Replace informal variations with formal equivalents"""
        text_lower = text.lower()
        
        # Replace variations
        for formal, variations in self.informal_variations.items():
            for variant in variations:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                text_lower = re.sub(pattern, formal, text_lower)
        
        # Remove filler words
        for filler in self.filler_words:
            pattern = r'\b' + re.escape(filler) + r'\b'
            text_lower = re.sub(pattern, '', text_lower)
        
        # Clean up extra spaces
        text_lower = " ".join(text_lower.split())
        
        return text_lower
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using Sentence-BERT"""
        
        # Encode texts
        embeddings = self.similarity_model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def _calculate_wer(self, hypothesis: str, reference: str) -> float:
        """Calculate Word Error Rate"""
        
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()
        
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        
        # Use Levenshtein distance for WER
        distance = Levenshtein.distance(hyp_words, ref_words)
        wer = distance / len(ref_words)
        
        return min(wer, 1.0)
    
    def _find_substitutions(self, 
                           hypothesis: str, 
                           reference: str) -> List[Tuple[str, str]]:
        """Find word-level substitutions"""
        
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()
        
        substitutions = []
        
        # Simple alignment based on position
        for i, (ref_word, hyp_word) in enumerate(zip(ref_words, hyp_words)):
            if ref_word != hyp_word:
                # Check if it's an informal variation
                is_variation = False
                for formal, variations in self.informal_variations.items():
                    if ref_word in formal.split() and hyp_word in variations:
                        is_variation = True
                        break
                
                if not is_variation:
                    substitutions.append((ref_word, hyp_word))
        
        return substitutions
    
    def _determine_validity(self, 
                           similarity: float, 
                           wer: float, 
                           is_informal: bool) -> bool:
        """Determine if the transcription is semantically valid"""
        
        # Exact or informal match
        if wer == 0.0 or is_informal:
            return True
        
        # High similarity and low WER
        if similarity >= 0.85 and wer <= 0.2:
            return True
        
        # Moderate similarity with very low WER
        if similarity >= 0.75 and wer <= 0.1:
            return True
        
        return False
    
    def _calculate_confidence(self, 
                            similarity: float, 
                            wer: float, 
                            is_informal: bool) -> float:
        """Calculate confidence score for the validation"""
        
        if wer == 0.0:
            return 1.0
        
        if is_informal:
            return 0.95
        
        # Weighted combination
        similarity_weight = 0.7
        wer_weight = 0.3
        
        wer_score = 1.0 - wer
        confidence = (similarity_weight * similarity + 
                     wer_weight * wer_score)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_feedback(self,
                          is_valid: bool,
                          similarity: float,
                          wer: float,
                          is_informal: bool,
                          substitutions: List[Tuple[str, str]]) -> str:
        """Generate human-readable feedback"""
        
        if wer == 0.0:
            return "Perfect! Every word matches exactly."
        
        if is_informal:
            return "Good communication! You used informal speech which is acceptable in casual contexts."
        
        if is_valid:
            if substitutions:
                subs_str = ", ".join([f"'{h}' for '{r}'" for r, h in substitutions[:3]])
                return f"Message clear despite minor differences: {subs_str}"
            else:
                return "Semantically correct with minor variations."
        
        # Not valid
        if similarity < 0.5:
            return "The meaning is significantly different from the target."
        elif wer > 0.5:
            return f"Too many word errors ({int(wer * 100)}% error rate)."
        else:
            return "Some important words were mispronounced or missed."

class BatchValidator:
    """
    Validate multiple audio samples efficiently
    """
    
    def __init__(self, validator: SemanticValidator):
        self.validator = validator
    
    def validate_batch(self, 
                       audio_list: List[np.ndarray],
                       target_list: List[str]) -> List[SemanticValidationResult]:
        """Validate multiple samples"""
        
        results = []
        for audio, target in zip(audio_list, target_list):
            result = self.validator.validate(audio, target)
            results.append(result)
        
        return results

# ============================================================================
# Usage Example
# ============================================================================

def demo():
    """Demo of the semantic validation system"""
    
    print("Initializing Semantic Validator...")
    validator = SemanticValidator(
        stt_model="base",  # or "tiny" for faster, "small" for better
        similarity_model="all-MiniLM-L6-v2"
    )
    
    # Example 1: Exact match
    print("\n" + "="*60)
    print("TEST 1: Exact match")
    # Simulate audio (in practice, load actual audio)
    audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1  # 3 seconds
    
    # For demo, we'll mock the transcription
    validator._transcribe = lambda a, s: "The quick brown fox jumps over the lazy dog"
    
    result = validator.validate(
        audio,
        "The quick brown fox jumps over the lazy dog"
    )
    
    print(f"Transcription: {result.transcription}")
    print(f"Valid: {result.is_semantically_valid}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"WER: {result.word_error_rate:.3f}")
    print(f"Feedback: {result.feedback}")
    
    # Example 2: Informal variation
    print("\n" + "="*60)
    print("TEST 2: Informal variation")
    
    validator._transcribe = lambda a, s: "I'm gonna go to the store"
    
    result = validator.validate(
        audio,
        "I'm going to go to the store"
    )
    
    print(f"Transcription: {result.transcription}")
    print(f"Valid: {result.is_semantically_valid}")
    print(f"Informal: {result.is_informal_variant}")
    print(f"Feedback: {result.feedback}")
    
    # Example 3: Minor errors
    print("\n" + "="*60)
    print("TEST 3: Minor errors")
    
    validator._transcribe = lambda a, s: "The quick brown fox jump over the lazy dog"
    
    result = validator.validate(
        audio,
        "The quick brown fox jumps over the lazy dog"
    )
    
    print(f"Transcription: {result.transcription}")
    print(f"Valid: {result.is_semantically_valid}")
    print(f"Substitutions: {result.substitutions}")
    print(f"Feedback: {result.feedback}")

if __name__ == "__main__":
    demo()