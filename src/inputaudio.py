import sounddevice as sd
from scipy.io.wavfile import write
def record_audio(duration=5, filename="audio.wav"):
    # Record audio from the microphone
    print("Recording...")
    
    # Check available input channels
    try:
        device_info = sd.query_devices(sd.default.device[0], 'input')
        channels = device_info['max_input_channels']
        print(f"Max input channels available: {channels}")
    except Exception as e:
        print(f"Error querying input device: {e}")
        channels = 1  # Fallback to mono if there's an error
    
    # Ensure we do not attempt to record with more channels than available
    if channels < 1:
        print("No input channels available. Exiting.")
        return
    
    audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, 44100, audio_data)  # Save as WAV file
    print(f"Audio saved as {filename}")

# Call the record_audio function
if __name__ == "__main__":
    record_audio(duration=5, filename="audio.wav")