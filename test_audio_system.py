"""
Quick audio system test - verify microphone and speakers work.
"""
import numpy as np
import sounddevice as sd
import time
import sys

print("\n" + "="*60)
print("🔊 AUDIO SYSTEM TEST")
print("="*60)

def test_speaker():
    """Test speaker with a simple beep."""
    print("\n1️⃣ Testing Speakers...")
    print("   You should hear a 1-second beep at 440 Hz (A note)")
    
    try:
        # Generate a 440 Hz sine wave (1 second)
        duration = 1.0
        sample_rate = 44100
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        print("   🔊 Playing beep...")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        print("   ✅ Speaker test complete")
        
        response = input("   Did you hear the beep? (y/n): ").lower()
        return response == 'y'
    except Exception as e:
        print(f"   ❌ Speaker test failed: {e}")
        return False

def test_microphone():
    """Test microphone by recording and playing back."""
    print("\n2️⃣ Testing Microphone...")
    print("   Recording 3 seconds - please speak...")
    
    try:
        duration = 3
        sample_rate = 16000
        
        print("   🎤 Recording...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("   ✅ Recording complete")
        
        # Check if we got any audio
        max_amplitude = np.max(np.abs(recording))
        print(f"   📊 Max amplitude: {max_amplitude:.4f}")
        
        if max_amplitude < 0.001:
            print("   ⚠️ Very low amplitude - microphone might not be working")
            return False
        
        print("   🔊 Playing back your recording...")
        sd.play(recording, samplerate=sample_rate)
        sd.wait()
        print("   ✅ Playback complete")
        
        response = input("   Did you hear your voice? (y/n): ").lower()
        return response == 'y'
    except Exception as e:
        print(f"   ❌ Microphone test failed: {e}")
        return False

def list_devices():
    """List available audio devices."""
    print("\n📋 Available Audio Devices:")
    print("-" * 60)
    
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
        
        status = " [DEFAULT]" if i == sd.default.device[0] else ""
        print(f"{i:2d}. {device['name']}")
        print(f"    Type: {', '.join(device_type)}{status}")
        print(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
        print(f"    Sample Rate: {device['default_samplerate']} Hz")
        print()

def main():
    """Run audio tests."""
    print("\n📋 Listing audio devices first...\n")
    list_devices()
    
    print("\n" + "="*60)
    print("Starting Audio Tests...")
    print("="*60)
    
    # Test speaker
    speaker_ok = test_speaker()
    
    # Test microphone
    mic_ok = test_microphone()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"   Speakers:    {'✅ Working' if speaker_ok else '❌ Not Working'}")
    print(f"   Microphone:  {'✅ Working' if mic_ok else '❌ Not Working'}")
    print("="*60)
    
    if speaker_ok and mic_ok:
        print("\n✅ All audio systems working! Ready for voice tests.")
        return 0
    else:
        print("\n⚠️ Some audio systems not working properly.")
        print("\nTroubleshooting:")
        if not speaker_ok:
            print("  • Check speaker/headphone connection")
            print("  • Check system volume settings")
            print("  • Try a different output device")
        if not mic_ok:
            print("  • Check microphone connection")
            print("  • Check microphone permissions in Windows Settings")
            print("  • Try a different input device")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test cancelled")
        sys.exit(1)
