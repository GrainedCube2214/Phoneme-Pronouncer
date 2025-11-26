#!/usr/bin/env python3
"""
Branch 2: STT + Semantic Validation with Live Recording
Autonomous semantic validation - just speak and get feedback!
Uses Whisper for transcription and semantic analysis
No target needed - validates if speech makes sense!
"""

import numpy as np
import whisper
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
import re
from dataclasses import dataclass
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from transformers import pipeline

# Recording settings
SAMPLE_RATE = 16000
DURATION_SECONDS = 5

# Color codes for terminal output
class C:
    G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; B = '\033[94m'
    RESET = '\033[0m'; BOLD = '\033[1m'

@dataclass
class SemanticAnalysisResult:
    """Result from autonomous semantic analysis"""
    transcription: str
    is_coherent: bool
    has_complete_thought: bool
    grammar_score: float
    fluency_score: float
    filler_word_count: int
    sentence_count: int
    word_count: int
    confidence: float
    feedback: str
    detected_issues: List[str]

class AutonomousSemanticAnalyzer:
    """
    Autonomous semantic analyzer - no target needed!
    Validates if speech is coherent, grammatical, and meaningful.
    """
    
    def __init__(self, 
                 stt_model: str = "base",
                 device: str = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Whisper for STT
        print(f"{C.Y}Loading Whisper model: {stt_model}...{C.RESET}")
        self.whisper = whisper.load_model(stt_model, device=self.device)
        
        # Load grammar checker (using a zero-shot classifier)
        print(f"{C.Y}Loading semantic analysis models...{C.RESET}")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
        except:
            print(f"{C.Y}Note: Using CPU for classification{C.RESET}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
        
        print(f"{C.G}✓ Models loaded!{C.RESET}\n")
        
        # Filler words to detect
        self.filler_words = {
            "uh", "um", "ah", "er", "like", "you know", 
            "well", "so", "basically", "literally", "actually",
            "kind of", "sort of", "i mean"
        }
        
        # Common informal contractions (for detection, not correction)
        self.informal_words = {
            "gonna", "wanna", "gotta", "kinda", "sorta",
            "dunno", "lemme", "gimme", "y'all", "ain't"
        }
        
    def analyze(self, audio_path: str) -> SemanticAnalysisResult:
        """
        Autonomous analysis pipeline - no target needed!
        """
        
        # Step 1: Transcribe audio
        transcription = self._transcribe(audio_path)
        
        if not transcription or len(transcription.strip()) < 3:
            return SemanticAnalysisResult(
                transcription=transcription,
                is_coherent=False,
                has_complete_thought=False,
                grammar_score=0.0,
                fluency_score=0.0,
                filler_word_count=0,
                sentence_count=0,
                word_count=0,
                confidence=0.0,
                feedback="No speech detected or transcription too short.",
                detected_issues=["No clear speech detected"]
            )
        
        # Step 2: Basic linguistic analysis
        word_count = len(transcription.split())
        sentence_count = self._count_sentences(transcription)
        filler_count = self._count_fillers(transcription)
        
        # Step 3: Coherence check
        is_coherent = self._check_coherence(transcription)
        
        # Step 4: Complete thought check
        has_complete_thought = self._has_complete_thought(transcription)
        
        # Step 5: Grammar/fluency scoring
        grammar_score = self._assess_grammar(transcription)
        fluency_score = self._assess_fluency(transcription, filler_count, word_count)
        
        # Step 6: Detect issues
        issues = self._detect_issues(transcription, filler_count, word_count, is_coherent)
        
        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(
            is_coherent, has_complete_thought, grammar_score, fluency_score
        )
        
        # Step 8: Generate feedback
        feedback = self._generate_feedback(
            is_coherent, has_complete_thought, grammar_score, 
            fluency_score, filler_count, issues
        )
        
        return SemanticAnalysisResult(
            transcription=transcription,
            is_coherent=is_coherent,
            has_complete_thought=has_complete_thought,
            grammar_score=grammar_score,
            fluency_score=fluency_score,
            filler_word_count=filler_count,
            sentence_count=sentence_count,
            word_count=word_count,
            confidence=confidence,
            feedback=feedback,
            detected_issues=issues
        )
    
    def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper"""
        result = self.whisper.transcribe(
            audio_path,
            language="en",
            fp16=(self.device == "cuda")
        )
        return result["text"].strip()
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def _count_fillers(self, text: str) -> int:
        """Count filler words"""
        text_lower = text.lower()
        count = 0
        for filler in self.filler_words:
            count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
        return count
    
    def _check_coherence(self, text: str) -> bool:
        """
        Check if text is coherent using zero-shot classification
        """
        if len(text.split()) < 3:
            return False
        
        try:
            # Check if text represents a coherent statement
            result = self.classifier(
                text,
                candidate_labels=[
                    "coherent statement",
                    "random words",
                    "incomplete thought",
                    "nonsense"
                ],
                multi_label=False
            )
            
            # If "coherent statement" is top label with good confidence
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            return top_label == "coherent statement" and top_score > 0.4
        
        except:
            # Fallback: basic heuristic
            words = text.split()
            return len(words) >= 3 and len(set(words)) >= 2
    
    def _has_complete_thought(self, text: str) -> bool:
        """Check if text expresses a complete thought"""
        
        # Must have reasonable length
        words = text.split()
        if len(words) < 3:
            return False
        
        # Must have some sentence structure (contains verb-like words)
        # Simple heuristic: check for common verb patterns
        text_lower = text.lower()
        verb_indicators = [
            ' is ', ' are ', ' was ', ' were ', ' be ', ' been ',
            ' have ', ' has ', ' had ', ' do ', ' does ', ' did ',
            ' can ', ' could ', ' will ', ' would ', ' should ',
            ' go ', ' goes ', ' went ', ' make ', ' makes ', ' made ',
            ' get ', ' gets ', ' got ', ' take ', ' takes ', ' took ',
            ' see ', ' saw ', ' come ', ' came ', ' think ', ' thought '
        ]
        
        has_verb = any(verb in text_lower for verb in verb_indicators)
        
        # Must end properly or have clause structure
        has_proper_ending = text.rstrip().endswith(('.', '!', '?')) or len(words) > 5
        
        return has_verb and has_proper_ending
    
    def _assess_grammar(self, text: str) -> float:
        """Assess grammatical quality (0-1 scale)"""
        
        score = 1.0
        
        # Penalize missing capitalization at start
        if text and not text[0].isupper():
            score -= 0.1
        
        # Check for basic sentence structure
        words = text.split()
        if len(words) < 3:
            score -= 0.3
        
        # Penalize excessive informal contractions
        informal_count = sum(1 for word in self.informal_words if word in text.lower())
        if informal_count > 0:
            score -= min(0.2, informal_count * 0.1)
        
        # Penalize repeated words
        word_list = [w.lower() for w in words]
        if len(word_list) != len(set(word_list)):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_fluency(self, text: str, filler_count: int, word_count: int) -> float:
        """Assess fluency (0-1 scale)"""
        
        if word_count == 0:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Penalize high filler ratio
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        score -= min(0.4, filler_ratio * 2)
        
        # Penalize very short utterances
        if word_count < 5:
            score -= 0.2
        
        # Penalize no sentence structure
        if not any(p in text for p in '.!?'):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _detect_issues(self, text: str, filler_count: int, 
                      word_count: int, is_coherent: bool) -> List[str]:
        """Detect specific issues with the speech"""
        
        issues = []
        
        if not is_coherent:
            issues.append("Speech lacks coherence")
        
        if word_count < 5:
            issues.append("Very short utterance")
        
        if filler_count > word_count * 0.2:
            issues.append(f"Excessive filler words ({filler_count})")
        
        if not any(p in text for p in '.!?'):
            issues.append("No clear sentence structure")
        
        words = text.split()
        if len(words) != len(set(w.lower() for w in words)) and len(words) > 3:
            issues.append("Word repetition detected")
        
        return issues
    
    def _calculate_confidence(self, is_coherent: bool, has_complete_thought: bool,
                            grammar_score: float, fluency_score: float) -> float:
        """Calculate overall confidence in the speech quality"""
        
        if not is_coherent or not has_complete_thought:
            return min(0.5, (grammar_score + fluency_score) / 2)
        
        # Weighted average
        confidence = (
            0.3 * (1.0 if is_coherent else 0.0) +
            0.3 * (1.0 if has_complete_thought else 0.0) +
            0.2 * grammar_score +
            0.2 * fluency_score
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_feedback(self, is_coherent: bool, has_complete_thought: bool,
                          grammar_score: float, fluency_score: float,
                          filler_count: int, issues: List[str]) -> str:
        """Generate human-readable feedback"""
        
        if is_coherent and has_complete_thought and grammar_score > 0.8 and fluency_score > 0.8:
            return "Excellent! Clear, coherent, and well-articulated speech."
        
        if is_coherent and has_complete_thought:
            if filler_count > 3:
                return "Good message, but try to reduce filler words for better fluency."
            else:
                return "Clear and understandable communication."
        
        if not is_coherent:
            return "Speech lacks coherence - try speaking in complete sentences."
        
        if not has_complete_thought:
            return "Incomplete thought - try expressing a full idea."
        
        if issues:
            return f"Issues detected: {', '.join(issues[:2])}"
        
        return "Needs improvement in clarity and structure."


def record_audio(duration=DURATION_SECONDS):
    """Record audio with normalization."""
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


def print_analysis_results(result: SemanticAnalysisResult):
    """Print results in a nice format."""
    
    print(f"{C.BOLD}{'='*70}{C.RESET}")
    print(f"{C.BOLD}SEMANTIC ANALYSIS RESULTS{C.RESET}")
    print(f"{C.BOLD}{'='*70}{C.RESET}\n")
    
    print(f"{C.B}You said:{C.RESET} \"{result.transcription}\"\n")
    
    # Overall quality score
    quality_score = result.confidence * 100
    
    if quality_score >= 80:
        color, emoji = C.G, "🌟"
        status = "EXCELLENT"
    elif quality_score >= 60:
        color, emoji = C.Y, "👍"
        status = "GOOD"
    else:
        color, emoji = C.R, "📚"
        status = "NEEDS WORK"
    
    print(f"{color}{emoji} Overall Quality: {quality_score:.0f}% - {status}{C.RESET}\n")
    
    # Detailed metrics
    print(f"{C.BOLD}Analysis:{C.RESET}")
    print(f"  Coherence:        {C.G}✓{C.RESET if result.is_coherent else C.R}✗{C.RESET} {'Yes' if result.is_coherent else 'No'}")
    print(f"  Complete Thought: {C.G}✓{C.RESET if result.has_complete_thought else C.R}✗{C.RESET} {'Yes' if result.has_complete_thought else 'No'}")
    print(f"  Grammar Score:    {result.grammar_score:.2%}")
    print(f"  Fluency Score:    {result.fluency_score:.2%}\n")
    
    print(f"{C.BOLD}Statistics:{C.RESET}")
    print(f"  Words:         {result.word_count}")
    print(f"  Sentences:     {result.sentence_count}")
    print(f"  Filler Words:  {result.filler_word_count}")
    
    if result.filler_word_count > 0:
        filler_pct = (result.filler_word_count / result.word_count * 100) if result.word_count > 0 else 0
        print(f"                 ({filler_pct:.1f}% of speech)")
    print()
    
    # Issues
    if result.detected_issues:
        print(f"{C.Y}{C.BOLD}Issues Detected:{C.RESET}")
        for issue in result.detected_issues:
            print(f"  • {issue}")
        print()
    else:
        print(f"{C.G}🎉 No issues detected!{C.RESET}\n")
    
    # Feedback
    print(f"{C.BOLD}Feedback:{C.RESET}")
    print(f"  {result.feedback}\n")
    
    print(f"{C.BOLD}{'='*70}{C.RESET}\n")


def main():
    """Main interactive loop."""
    
    print(f"{C.BOLD}AUTONOMOUS SEMANTIC VALIDATOR (Branch 2){C.RESET}\n")
    print("How it works:")
    print("  • Just speak naturally")
    print("  • STT transcribes what you said")
    print("  • Semantic analysis validates meaning")
    print("  • Get feedback on clarity and coherence!\n")
    
    # Initialize analyzer
    analyzer = AutonomousSemanticAnalyzer(
        stt_model="base"  # Use "tiny" for faster, "small"/"medium" for better
    )
    
    while True:
        cmd = input(f"{C.B}Press Enter to record (or 'quit'): {C.RESET}")
        
        if cmd.lower() in ['q', 'quit', 'exit']:
            print(f"\n{C.G}Goodbye!{C.RESET}")
            break
        
        # Record
        audio = record_audio()
        
        # Save to temporary file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(temp_wav, audio, SAMPLE_RATE)
        
        try:
            # Analyze
            print(f"{C.Y}Analyzing transcription and semantics...{C.RESET}\n")
            result = analyzer.analyze(temp_wav)
            
            # Display results
            print_analysis_results(result)
            
        except Exception as e:
            print(f"{C.R}Error during analysis: {e}{C.RESET}\n")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        
        print(f"{C.Y}{'─'*70}{C.RESET}\n")


if __name__ == "__main__":
    main()