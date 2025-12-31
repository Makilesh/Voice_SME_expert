"""Test audio device functionality."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio import MicrophoneCapture, VirtualAudioCapture
import numpy as np
import time


def list_devices():
    """List all available audio devices."""
    print("\n=== Available Microphone Devices ===")
    devices = MicrophoneCapture.list_devices()
    
    if not devices:
        print("No microphone devices found")
        return
    
    for device in devices:
        print(f"[{device['index']}] {device['name']}")
        print(f"    Channels: {device['channels']}, Sample Rate: {device['sample_rate']:.0f}Hz")
        print(f"    Host API: {device['hostapi']}")
        print()
    
    print("\n=== Available Loopback Devices ===")
    loopback_devices = VirtualAudioCapture.list_loopback_devices()
    
    if not loopback_devices:
        print("No loopback devices found")
        print("For virtual audio capture, you may need to install:")
        print("  - Windows: VB-Audio Virtual Cable or enable Stereo Mix")
        print("  - macOS: BlackHole or Soundflower")
        print("  - Linux: PulseAudio loopback module")
        return
    
    for device in loopback_devices:
        print(f"[{device['index']}] {device['name']}")
        print(f"    Channels: {device['channels']}, Sample Rate: {device['sample_rate']:.0f}Hz")
        print()


def test_device(device_index: int, duration: int = 5):
    """Test recording from a specific device."""
    print(f"\nTesting device {device_index} for {duration} seconds...")
    print("Speak into the microphone...\n")
    
    try:
        # Initialize capture
        capture = MicrophoneCapture(device_index=device_index)
        
        # Capture stats
        chunks = []
        start_time = time.time()
        
        def callback(audio_chunk: np.ndarray):
            chunks.append(audio_chunk)
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            db = 20 * np.log10(rms) if rms > 0 else -100
            
            # Print level meter
            bars = int((db + 60) / 3)  # Scale from -60dB to 0dB
            meter = "█" * max(0, bars) + "░" * max(0, 20 - bars)
            print(f"\r[{meter}] {db:6.1f} dB", end="", flush=True)
        
        # Start capture
        capture.start_capture(callback)
        
        # Run for specified duration
        time.sleep(duration)
        
        # Stop capture
        capture.stop_capture()
        
        # Print stats
        print(f"\n\nCapture complete!")
        print(f"Captured {len(chunks)} chunks")
        print(f"Total samples: {sum(len(c) for c in chunks)}")
        print(f"Duration: {time.time() - start_time:.2f}s")
        
        device_info = capture.get_device_info()
        if device_info:
            print(f"\nDevice Info:")
            print(f"  Name: {device_info['name']}")
            print(f"  Channels: {device_info['channels']}")
            print(f"  Sample Rate: {device_info['sample_rate']:.0f}Hz")
    
    except Exception as e:
        print(f"\nError testing device: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="Test audio devices")
    parser.add_argument("--list", action="store_true", help="List available devices")
    parser.add_argument("--device", type=int, help="Device index to test")
    parser.add_argument("--test", action="store_true", help="Test the device")
    parser.add_argument("--duration", type=int, default=5, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    if args.list:
        list_devices()
    elif args.device is not None and args.test:
        test_device(args.device, args.duration)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
