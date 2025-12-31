#!/usr/bin/env python3
"""Wake word training script."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train custom wake words for voice activation"
    )
    
    parser.add_argument(
        "wake_word",
        help="Wake word phrase to train"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/wake_words",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--positive-samples",
        type=str,
        help="Directory containing positive samples (wake word recordings)"
    )
    
    parser.add_argument(
        "--negative-samples",
        type=str,
        help="Directory containing negative samples (background audio)"
    )
    
    parser.add_argument(
        "--record-samples",
        type=int,
        default=0,
        help="Number of samples to record interactively"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID for recording"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def record_samples(wake_word: str, num_samples: int, device_id: int, output_dir: Path):
    """Record positive samples interactively."""
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    
    sample_rate = 16000
    duration = 2.0  # seconds per sample
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📢 Recording {num_samples} samples of: '{wake_word}'")
    print("Press Enter before each recording.\n")
    
    for i in range(num_samples):
        input(f"Sample {i+1}/{num_samples} - Press Enter and say '{wake_word}'...")
        
        print("🎤 Recording...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
            dtype=np.float32
        )
        sd.wait()
        
        # Save sample
        sample_path = output_dir / f"sample_{i+1:03d}.wav"
        sf.write(sample_path, audio, sample_rate)
        
        print(f"✅ Saved: {sample_path}")
    
    print(f"\n✅ Recorded {num_samples} samples to {output_dir}")
    return output_dir


def train_wake_word(
    wake_word: str,
    positive_dir: Path,
    negative_dir: Path,
    output_dir: Path
):
    """Train wake word model using openwakeword."""
    try:
        # Note: openwakeword training requires additional setup
        # This is a simplified implementation
        
        print(f"\n🔧 Training wake word model for: '{wake_word}'")
        print(f"  Positive samples: {positive_dir}")
        print(f"  Negative samples: {negative_dir}")
        print(f"  Output: {output_dir}")
        
        # Check for positive samples
        if not positive_dir.exists():
            print(f"❌ Positive samples directory not found: {positive_dir}")
            return False
        
        positive_files = list(positive_dir.glob("*.wav"))
        if not positive_files:
            print(f"❌ No .wav files found in {positive_dir}")
            return False
        
        print(f"  Found {len(positive_files)} positive samples")
        
        # Check for negative samples
        negative_files = []
        if negative_dir and negative_dir.exists():
            negative_files = list(negative_dir.glob("*.wav"))
            print(f"  Found {len(negative_files)} negative samples")
        
        # Training would happen here
        # openwakeword uses a specific training pipeline
        
        print("\n⚠️  Note: Full wake word training requires:")
        print("   1. Many positive samples (50+)")
        print("   2. Diverse negative samples")
        print("   3. openwakeword training infrastructure")
        print("\n   For custom wake words, consider:")
        print("   - Using the openwakeword training scripts")
        print("   - Fine-tuning existing models")
        print("   - Using the built-in wake words as alternatives")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Wake Word: {wake_word}\n")
            f.write(f"Positive Samples: {len(positive_files)}\n")
            f.write(f"Negative Samples: {len(negative_files)}\n")
        
        print(f"\n📝 Metadata saved to: {metadata_path}")
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False


def list_builtin_wake_words():
    """List available built-in wake words."""
    try:
        from openwakeword.model import Model
        
        print("\n📋 Built-in Wake Words (openwakeword):")
        print("-" * 40)
        
        builtin = [
            "hey_jarvis",
            "alexa", 
            "hey_mycroft",
            "hey_rhasspy"
        ]
        
        for word in builtin:
            print(f"  • {word}")
        
        print("\nThese can be used directly without training.")
        
    except ImportError:
        print("⚠️  openwakeword not installed")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Load config
    config = load_config()
    
    output_dir = Path(args.output_dir)
    wake_word_dir = output_dir / args.wake_word.lower().replace(" ", "_")
    
    # Show built-in options
    list_builtin_wake_words()
    
    # Record samples if requested
    if args.record_samples > 0:
        positive_dir = record_samples(
            args.wake_word,
            args.record_samples,
            args.device,
            wake_word_dir / "positive"
        )
    else:
        positive_dir = Path(args.positive_samples) if args.positive_samples else None
    
    # Get negative samples directory
    negative_dir = Path(args.negative_samples) if args.negative_samples else None
    
    # Train if we have samples
    if positive_dir and positive_dir.exists():
        train_wake_word(
            args.wake_word,
            positive_dir,
            negative_dir,
            wake_word_dir
        )
    elif not args.record_samples:
        print("\n⚠️  No samples provided. Use --record-samples or --positive-samples")


if __name__ == "__main__":
    main()
