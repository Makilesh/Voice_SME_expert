#!/usr/bin/env python3
"""Speaker enrollment script for voice identification."""
import argparse
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.diarization import VoiceEnrollment, SpeakerDiarizer
from src.audio import MicrophoneCapture
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enroll a speaker's voice for identification"
    )
    
    parser.add_argument(
        "name",
        help="Name of the speaker to enroll"
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Audio file to use for enrollment (optional)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10)"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID"
    )
    
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="./data/speaker_profiles",
        help="Directory to store speaker profiles"
    )
    
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List all enrolled speakers"
    )
    
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a speaker profile by name"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


async def record_audio(device_id: int, duration: float, sample_rate: int = 16000):
    """Record audio from microphone."""
    import numpy as np
    
    print(f"\nRecording for {duration} seconds...")
    print("Please speak clearly into the microphone.")
    print("Say a few sentences to provide a good voice sample.\n")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        await asyncio.sleep(1)
    
    print("🎤 Recording NOW!")
    
    # Create capture
    capture = MicrophoneCapture(
        device_id=device_id,
        sample_rate=sample_rate
    )
    
    await capture.start_capture()
    
    # Collect audio
    frames = []
    samples_needed = int(duration * sample_rate)
    samples_collected = 0
    
    while samples_collected < samples_needed:
        audio = await capture.read_audio(chunk_size=1024)
        if audio is not None:
            frames.append(audio)
            samples_collected += len(audio)
            
            # Progress indicator
            progress = min(100, int(samples_collected / samples_needed * 100))
            print(f"\rProgress: {progress}% ", end="")
    
    await capture.stop_capture()
    
    print("\n✅ Recording complete!")
    
    # Combine frames
    return np.concatenate(frames)


async def enroll_from_file(
    enrollment: VoiceEnrollment,
    name: str,
    audio_path: str
):
    """Enroll speaker from audio file."""
    import numpy as np
    import soundfile as sf
    
    print(f"\nLoading audio from: {audio_path}")
    
    # Load audio file
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != 16000:
        import scipy.signal
        samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, samples)
    
    # Ensure float32
    audio = audio.astype(np.float32)
    
    # Enroll
    success = await enrollment.enroll_speaker(name, audio)
    
    if success:
        print(f"✅ Successfully enrolled speaker: {name}")
    else:
        print(f"❌ Failed to enroll speaker: {name}")
    
    return success


async def enroll_from_microphone(
    enrollment: VoiceEnrollment,
    name: str,
    duration: float,
    device_id: int
):
    """Enroll speaker from microphone recording."""
    # Record audio
    audio = await record_audio(device_id, duration)
    
    # Enroll
    success = await enrollment.enroll_speaker(name, audio)
    
    if success:
        print(f"✅ Successfully enrolled speaker: {name}")
    else:
        print(f"❌ Failed to enroll speaker: {name}")
    
    return success


def list_enrolled_speakers(enrollment: VoiceEnrollment):
    """List all enrolled speakers."""
    speakers = enrollment.list_speakers()
    
    if not speakers:
        print("\nNo speakers enrolled yet.")
        return
    
    print(f"\n📋 Enrolled Speakers ({len(speakers)}):")
    print("-" * 40)
    
    for speaker in speakers:
        info = enrollment.get_speaker_info(speaker)
        if info:
            samples = info.get('sample_count', 'N/A')
            enrolled = info.get('enrolled_at', 'N/A')
            print(f"  • {speaker}")
            print(f"    Samples: {samples}")
            print(f"    Enrolled: {enrolled}")
        else:
            print(f"  • {speaker}")
    
    print()


def delete_speaker(enrollment: VoiceEnrollment, name: str):
    """Delete a speaker profile."""
    if enrollment.delete_speaker(name):
        print(f"✅ Deleted speaker profile: {name}")
    else:
        print(f"❌ Speaker not found: {name}")


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Load config
    config = load_config()
    
    # Initialize enrollment
    enrollment = VoiceEnrollment(profiles_directory=args.profiles_dir)
    
    # Handle list command
    if args.list_speakers:
        list_enrolled_speakers(enrollment)
        return
    
    # Handle delete command
    if args.delete:
        delete_speaker(enrollment, args.delete)
        return
    
    # Enroll speaker
    if args.audio_file:
        await enroll_from_file(enrollment, args.name, args.audio_file)
    else:
        await enroll_from_microphone(
            enrollment,
            args.name,
            args.duration,
            args.device
        )
    
    # Show all speakers
    list_enrolled_speakers(enrollment)


if __name__ == "__main__":
    asyncio.run(main())
