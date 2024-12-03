import torchaudio
import whisper
import soundfile as sf
import tempfile
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore", category = UserWarning)

def load_audio(audio_file):
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    return waveform, 16000

def transcribe_audio(audio_data):
    # Create a temporary file to store the audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        # Ensure audio_data is in the correct format
        if isinstance(audio_data, torch.Tensor):
            # Convert stereo to mono if necessary
            if audio_data.dim() == 2 and audio_data.shape[0] > 1:
                audio_data = torch.mean(audio_data, dim=0)
            audio_data = audio_data.squeeze().numpy()

        # Ensure the data is 2D (samples x channels)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        
        # Normalize audio if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()

        # Write the audio data to the temporary file
        sf.write(temp_audio_file.name, audio_data, 16000, format='WAV')
        
        # Load the Whisper model
        model = whisper.load_model("base")
        
        # Transcribe the audio using Whisper
        result = model.transcribe(temp_audio_file.name)
        predicted_text = result["text"]
        language = result.get("language", None)

    return predicted_text, language

# Example usage
if __name__ == "__main__":
    audio_file_path = "./temp_audio.wav"  # Replace with your audio file path
    audio_data, sample_rate = load_audio(audio_file_path)
    transcription, language = transcribe_audio(audio_data)

    print("Transcription:", transcription)
    print("Detected Language:", language)