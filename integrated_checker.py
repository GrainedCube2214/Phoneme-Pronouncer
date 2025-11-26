#!/usr/bin/env python3
"""
Integrated Pronunciation Analysis System
Combines Branch 1 (Phoneme Detection) and Branch 2 (Semantic Validation)
Records once, analyzes from both perspectives, gives comprehensive feedback
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
from typing import Dict, Any

# Import Branch 1 components
from live_checker_nomfa import (
    detect_phonemes_sliding_window,
    SAMPLE_RATE,
    DURATION_SECONDS
)

# Import Branch 2 components
from live_tts_val import (
    AutonomousSemanticAnalyzer,
    SemanticAnalysisResult
)

# Color codes
class C:
    G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; B = '\033[94m'
    C = '\033[96m'; M = '\033[95m'  # Cyan and Magenta for integration
    RESET = '\033[0m'; BOLD = '\033[1m'


class IntegratedPronunciationAnalyzer:
    """
    Combines phoneme-level and semantic-level analysis
    for comprehensive pronunciation feedback
    """
    
    def __init__(self):
        print(f"{C.BOLD}{C.C}{'='*70}{C.RESET}")
        print(f"{C.BOLD}{C.C}INTEGRATED PRONUNCIATION ANALYSIS SYSTEM{C.RESET}")
        print(f"{C.BOLD}{C.C}{'='*70}{C.RESET}\n")
        
        print(f"{C.Y}Initializing Branch 2 (Semantic Analyzer)...{C.RESET}")
        self.semantic_analyzer = AutonomousSemanticAnalyzer(stt_model="base")
        print(f"{C.G}✓ Branch 2 ready!{C.RESET}\n")
        
        print(f"{C.Y}Branch 1 (Phoneme Detector) will load on first use{C.RESET}\n")
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Run both branches on the same audio file
        """
        print(f"{C.C}{C.BOLD}Running Dual Analysis...{C.RESET}\n")
        
        # Branch 1: Phoneme Detection
        print(f"{C.Y}[Branch 1] Detecting phonemes...{C.RESET}")
        try:
            phoneme_results = detect_phonemes_sliding_window(audio_path)
        except Exception as e:
            print(f"{C.R}Branch 1 error: {e}{C.RESET}")
            phoneme_results = {'error': str(e)}
        
        # Branch 2: Semantic Validation
        print(f"{C.Y}[Branch 2] Analyzing semantics...{C.RESET}\n")
        try:
            semantic_results = self.semantic_analyzer.analyze(audio_path)
        except Exception as e:
            print(f"{C.R}Branch 2 error: {e}{C.RESET}")
            semantic_results = None
        
        return {
            'branch1_phonemes': phoneme_results,
            'branch2_semantics': semantic_results
        }
    
    def compare_and_synthesize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results from both branches and synthesize feedback
        """
        branch1 = results['branch1_phonemes']
        branch2 = results['branch2_semantics']
        
        # Extract key metrics
        synthesis = {
            'phoneme_clarity': 0.0,
            'semantic_quality': 0.0,
            'overall_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'alignment_score': 0.0
        }
        
        # Branch 1 metrics
        if 'error' not in branch1:
            n_phonemes = branch1.get('n_phonemes', 0)
            high_conf = branch1.get('high_confidence_count', 0)
            phoneme_clarity = (high_conf / n_phonemes * 100) if n_phonemes > 0 else 0
            synthesis['phoneme_clarity'] = phoneme_clarity
            
            # Assess phoneme quality
            if phoneme_clarity >= 80:
                synthesis['strengths'].append("Excellent phoneme clarity")
            elif phoneme_clarity >= 60:
                synthesis['strengths'].append("Good phoneme pronunciation")
            else:
                synthesis['weaknesses'].append("Many unclear phonemes detected")
        else:
            synthesis['weaknesses'].append("Phoneme detection failed")
        
        # Branch 2 metrics
        if branch2:
            semantic_quality = branch2.confidence * 100
            synthesis['semantic_quality'] = semantic_quality
            
            # Assess semantic quality
            if branch2.is_coherent and branch2.has_complete_thought:
                synthesis['strengths'].append("Clear and coherent message")
            else:
                synthesis['weaknesses'].append("Message lacks coherence")
            
            if branch2.grammar_score > 0.8:
                synthesis['strengths'].append("Good grammar structure")
            elif branch2.grammar_score < 0.5:
                synthesis['weaknesses'].append("Grammar needs improvement")
            
            if branch2.fluency_score > 0.8:
                synthesis['strengths'].append("Fluent delivery")
            elif branch2.fluency_score < 0.5:
                synthesis['weaknesses'].append("Fluency needs work")
            
            if branch2.filler_word_count > branch2.word_count * 0.15:
                synthesis['weaknesses'].append("Too many filler words")
        else:
            synthesis['weaknesses'].append("Semantic analysis failed")
        
        # Calculate alignment score (do phoneme clarity and semantic quality match?)
        if 'error' not in branch1 and branch2:
            phoneme_score = synthesis['phoneme_clarity'] / 100
            semantic_score = synthesis['semantic_quality'] / 100
            
            # High alignment = both good or both bad
            # Low alignment = mismatch (e.g., clear phonemes but incoherent message)
            alignment = 1.0 - abs(phoneme_score - semantic_score)
            synthesis['alignment_score'] = alignment * 100
            
            if alignment > 0.8:
                synthesis['strengths'].append("Consistent quality across phoneme and semantic levels")
            elif alignment < 0.5:
                synthesis['weaknesses'].append("Mismatch between pronunciation and meaning clarity")
        
        # Calculate overall score
        phoneme_weight = 0.4
        semantic_weight = 0.4
        alignment_weight = 0.2
        
        overall = (
            phoneme_weight * synthesis['phoneme_clarity'] +
            semantic_weight * synthesis['semantic_quality'] +
            alignment_weight * synthesis['alignment_score']
        )
        synthesis['overall_score'] = overall
        
        # Generate recommendations
        if synthesis['phoneme_clarity'] < 60:
            synthesis['recommendations'].append(
                "Focus on pronouncing individual sounds more clearly"
            )
        
        if branch2 and not branch2.is_coherent:
            synthesis['recommendations'].append(
                "Work on structuring your thoughts before speaking"
            )
        
        if branch2 and branch2.filler_word_count > 3:
            synthesis['recommendations'].append(
                "Practice reducing filler words (um, uh, like, etc.)"
            )
        
        if synthesis['alignment_score'] < 50:
            synthesis['recommendations'].append(
                "Work on connecting clear pronunciation with coherent expression"
            )
        
        if not synthesis['recommendations']:
            synthesis['recommendations'].append(
                "Excellent work! Keep practicing to maintain this level"
            )
        
        return synthesis


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


def print_branch1_summary(results: Dict[str, Any]):
    """Print condensed Branch 1 results"""
    
    print(f"{C.BOLD}{C.B}{'─'*70}{C.RESET}")
    print(f"{C.BOLD}{C.B}BRANCH 1: PHONEME ANALYSIS{C.RESET}")
    print(f"{C.BOLD}{C.B}{'─'*70}{C.RESET}\n")
    
    if 'error' in results:
        print(f"{C.R}✗ Error: {results['error']}{C.RESET}\n")
        return
    
    transcript = results.get('transcript', 'N/A')
    n_phonemes = results.get('n_phonemes', 0)
    high_conf = results.get('high_confidence_count', 0)
    avg_conf = results.get('avg_confidence', 0)
    phoneme_seq = results.get('phoneme_sequence', '')
    
    clarity_pct = (high_conf / n_phonemes * 100) if n_phonemes > 0 else 0
    
    print(f"{C.B}Transcription:{C.RESET} \"{transcript}\"")
    print(f"{C.B}Phonemes Detected:{C.RESET} {n_phonemes}")
    print(f"{C.B}Clear Phonemes:{C.RESET} {high_conf}/{n_phonemes} ({clarity_pct:.0f}%)")
    print(f"{C.B}Avg Confidence:{C.RESET} {avg_conf:.2f}")
    print(f"{C.B}Sequence:{C.RESET} {phoneme_seq}\n")


def print_branch2_summary(results: SemanticAnalysisResult):
    """Print condensed Branch 2 results"""
    
    print(f"{C.BOLD}{C.G}{'─'*70}{C.RESET}")
    print(f"{C.BOLD}{C.G}BRANCH 2: SEMANTIC ANALYSIS{C.RESET}")
    print(f"{C.BOLD}{C.G}{'─'*70}{C.RESET}\n")
    
    if not results:
        print(f"{C.R}✗ Analysis failed{C.RESET}\n")
        return
    
    print(f"{C.G}Transcription:{C.RESET} \"{results.transcription}\"")
    print(f"{C.G}Coherent:{C.RESET} {'Yes ✓' if results.is_coherent else 'No ✗'}")
    print(f"{C.G}Complete Thought:{C.RESET} {'Yes ✓' if results.has_complete_thought else 'No ✗'}")
    print(f"{C.G}Grammar Score:{C.RESET} {results.grammar_score:.1%}")
    print(f"{C.G}Fluency Score:{C.RESET} {results.fluency_score:.1%}")
    print(f"{C.G}Confidence:{C.RESET} {results.confidence:.1%}")
    print(f"{C.G}Words/Sentences:{C.RESET} {results.word_count} words, {results.sentence_count} sentences")
    
    if results.filler_word_count > 0:
        print(f"{C.Y}Filler Words:{C.RESET} {results.filler_word_count}")
    
    print()


def print_integrated_results(synthesis: Dict[str, Any]):
    """Print the integrated analysis and final verdict"""
    
    print(f"{C.BOLD}{C.M}{'='*70}{C.RESET}")
    print(f"{C.BOLD}{C.M}INTEGRATED ANALYSIS & FINAL VERDICT{C.RESET}")
    print(f"{C.BOLD}{C.M}{'='*70}{C.RESET}\n")
    
    # Overall score with visual indicator
    overall = synthesis['overall_score']
    
    if overall >= 80:
        color, emoji, rating = C.G, "🌟", "EXCELLENT"
    elif overall >= 65:
        color, emoji, rating = C.Y, "👍", "GOOD"
    elif overall >= 50:
        color, emoji, rating = C.Y, "📈", "FAIR"
    else:
        color, emoji, rating = C.R, "📚", "NEEDS IMPROVEMENT"
    
    print(f"{color}{C.BOLD}{emoji} OVERALL SCORE: {overall:.1f}/100 - {rating}{C.RESET}\n")
    
    # Component scores
    print(f"{C.BOLD}Component Scores:{C.RESET}")
    print(f"  Phoneme Clarity:    {synthesis['phoneme_clarity']:.1f}/100")
    print(f"  Semantic Quality:   {synthesis['semantic_quality']:.1f}/100")
    print(f"  Alignment:          {synthesis['alignment_score']:.1f}/100")
    print()
    
    # Strengths
    if synthesis['strengths']:
        print(f"{C.G}{C.BOLD}✓ Strengths:{C.RESET}")
        for strength in synthesis['strengths']:
            print(f"  • {strength}")
        print()
    
    # Weaknesses
    if synthesis['weaknesses']:
        print(f"{C.R}{C.BOLD}✗ Areas for Improvement:{C.RESET}")
        for weakness in synthesis['weaknesses']:
            print(f"  • {weakness}")
        print()
    
    # Recommendations
    print(f"{C.C}{C.BOLD}💡 Recommendations:{C.RESET}")
    for rec in synthesis['recommendations']:
        print(f"  • {rec}")
    print()
    
    # Interpretation guide
    print(f"{C.BOLD}What this means:{C.RESET}")
    if overall >= 80:
        print(f"  Your pronunciation is excellent at both the phoneme and semantic")
        print(f"  levels. You're speaking clearly and coherently.")
    elif overall >= 65:
        print(f"  Good job! Your speech is generally clear, but there's room for")
        print(f"  improvement in specific areas identified above.")
    elif overall >= 50:
        print(f"  You're making progress, but need to work on either pronunciation")
        print(f"  clarity or message coherence (or both).")
    else:
        print(f"  Significant improvement needed. Focus on the recommendations above")
        print(f"  and practice regularly.")
    
    print(f"\n{C.BOLD}{C.M}{'='*70}{C.RESET}\n")


def main():
    """Main integrated analysis loop"""
    
    print(f"\n{C.BOLD}{C.C}DUAL-BRANCH PRONUNCIATION ANALYSIS SYSTEM{C.RESET}\n")
    print("This system analyzes your speech from two perspectives:")
    print(f"  {C.B}Branch 1:{C.RESET} Phoneme-level detection (what sounds you make)")
    print(f"  {C.G}Branch 2:{C.RESET} Semantic validation (if your message makes sense)")
    print(f"  {C.M}Integration:{C.RESET} Combined analysis and recommendations\n")
    
    # Initialize
    analyzer = IntegratedPronunciationAnalyzer()
    
    while True:
        print(f"{C.C}{'─'*70}{C.RESET}\n")
        cmd = input(f"{C.C}Press Enter to record (or 'quit'): {C.RESET}")
        
        if cmd.lower() in ['q', 'quit', 'exit']:
            print(f"\n{C.G}Goodbye!{C.RESET}")
            break
        
        # Record audio
        audio = record_audio()
        
        # Save to temporary file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(temp_wav, audio, SAMPLE_RATE)
        
        try:
            # Run both branches
            results = analyzer.analyze_audio(temp_wav)
            
            # Print individual branch results
            print_branch1_summary(results['branch1_phonemes'])
            print_branch2_summary(results['branch2_semantics'])
            
            # Synthesize and print integrated results
            synthesis = analyzer.compare_and_synthesize(results)
            print_integrated_results(synthesis)
            
        except Exception as e:
            print(f"{C.R}Error during analysis: {e}{C.RESET}\n")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)


if __name__ == "__main__":
    main()