import numpy as np
import librosa
import tensorflow.keras.models as models
import joblib
import warnings
import os
from tensorflow.keras.utils import to_categorical

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import necessary functions
from evaluation.wav2vec import Wav2VecTranscriber, LanguageDetector, load_audio
from evaluation.bertanalysis import analyze_sentiment

# Define emotion mapping
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """Extract audio features to match the expected input shape (275, 99)."""
    try:
        # Load and resample audio
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Extract different types of features
        # MFCC (40 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        
        # Mel spectrogram (40 features)
        mel_spect = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=40)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=16000)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000)
        
        # Zero crossing rate
        zero_crossing = librosa.feature.zero_crossing_rate(audio)
        
        # Chroma features (12 features)
        chroma = librosa.feature.chroma_stft(y=audio, sr=16000)
        
        # Stack all features
        combined_features = np.vstack([
            mfccs,  # 40 features
            mel_spect,  # 40 features
            spectral_centroids,  # 1 feature
            spectral_rolloff,  # 1 feature
            spectral_bandwidth,  # 1 feature
            zero_crossing,  # 1 feature
            chroma  # 12 features
            # Total: 96 features + 3 padding = 99 features
        ])
        
        # Add padding to reach 99 features
        padding = np.zeros((3, combined_features.shape[1]))
        combined_features = np.vstack([combined_features, padding])
        
        return combined_features.T
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def pad_or_truncate_features(features, target_length=275):
    """Pad or truncate features to match expected length of 275."""
    if features is None:
        return None
        
    if features.shape[0] > target_length:
        return features[:target_length, :]
    else:
        pad_width = ((0, target_length - features.shape[0]), (0, 0))
        return np.pad(features, pad_width, mode='constant')

def extract_metadata_features(file_path):
    """Extract metadata features from filename."""
    try:
        # Parse filename (assuming RAVDESS filename format)
        filename = os.path.basename(file_path)
        parts = filename.replace('.wav', '').split('-')
        
        # Create one-hot encoded features
        modality = to_categorical(int(parts[0]) - 1, num_classes=3)
        vocal_channel = to_categorical(int(parts[1]) - 1, num_classes=2)
        intensity = to_categorical(int(parts[3]) - 1, num_classes=2)
        statement = to_categorical(int(parts[4]) - 1, num_classes=2)
        repetition = to_categorical(int(parts[5]) - 1, num_classes=2)
        actor_num = int(parts[6])
        gender = [1, 0] if actor_num % 2 != 0 else [0, 1]
        actor = to_categorical(actor_num - 1, num_classes=24)
        
        # Combine all metadata features
        return np.concatenate([
            modality, vocal_channel, intensity, statement, 
            repetition, gender, actor
        ])
    except Exception as e:
        # If filename parsing fails, return zeros (neutral values)
        print(f"Warning: Could not parse filename for metadata, using neutral values: {e}")
        return np.zeros(37)  # Total size of all metadata features

def predict_emotion(audio_file_path, model, label_encoder):
    """Predict emotion for a single audio file"""
    try:
        # Extract and process features
        features = extract_features(audio_file_path)
        if features is None:
            raise Exception("Failed to extract audio features")
        
        # Debug prints
        print(f"Initial feature shape: {features.shape}")
        
        # Pad or truncate
        features_padded = pad_or_truncate_features(features)
        if features_padded is None:
            raise Exception("Failed to pad features")
            
        print(f"Padded feature shape: {features_padded.shape}")
        
        # Reshape for model input
        features_reshaped = features_padded.reshape(1, 275, 99)
        print(f"Final feature shape: {features_reshaped.shape}")
        
        # Create metadata features (neutral values for non-RAVDESS files)
        metadata_features = np.zeros(37)
        metadata_reshaped = metadata_features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(
            [features_reshaped, metadata_reshaped],
            verbose=0
        )
        
        # Get emotion
        emotion_index = np.argmax(prediction)
        emotion_code = label_encoder.inverse_transform([emotion_index])[0]
        emotion_word = EMOTION_LABELS.get(emotion_code, f"Unknown emotion code: {emotion_code}")
        
        return emotion_word
        
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        raise e

def main():
    # Initialize Wav2Vec models
    transcriber = Wav2VecTranscriber()
    lang_detector = LanguageDetector()
    
    # Paths to emotion model and label encoder
    MODEL_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/emotion_model.keras"
    ENCODER_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/label_encoder.joblib"

    # Specify the path to your audio file
    audio_file_path = "temp_audio.wav"  # Replace with your actual audio file path

    # Load the trained emotion model and label encoder
    try:
        import tensorflow.keras as keras
        model = keras.models.load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"Error loading model or label encoder: {e}")
        return

    # Load the audio data
    audio_data, sample_rate = load_audio(audio_file_path)

    # Detect language using Wav2Vec
    language = lang_detector.detect_language(audio_data)
    print(f"Detected Language: {language}")

    # Transcribe audio using Wav2Vec
    transcription = transcriber.transcribe(audio_data)
    print(f"Transcription: {transcription}")



    # Predict emotion from the audio
    try:
        emotion = predict_emotion(audio_file_path, model, label_encoder)
        print(f"Predicted Emotion: {emotion}")
    except Exception as e:
        print(f"Error predicting emotion: {e}")

if __name__ == "__main__":
    main()