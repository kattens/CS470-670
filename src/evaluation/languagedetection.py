from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import librosa
import numpy as np
import os
from evaluation.wav2vec import Wav2VecTranscriber
# Define cache directory
CACHE_DIR = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/model_cache"

class LanguageDetector:
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53-language-identification"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = os.path.join(CACHE_DIR, model_name.replace('/', '_'))
        
        # Language mapping
        self.id2label = {
            0: "Arabic", 1: "Czech", 2: "German", 3: "English", 4: "Spanish",
            5: "Persian", 6: "French", 7: "Hindi", 8: "Italian", 9: "Japanese",
            10: "Korean", 11: "Dutch", 12: "Polish", 13: "Portuguese", 14: "Russian",
            15: "Turkish", 16: "Vietnamese", 17: "Chinese", 18: "Telugu", 19: "Urdu"
        }
        
        if not os.path.exists(self.model_dir):
            print(f"Downloading language detection model to {self.model_dir}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name, cache_dir=CACHE_DIR)
            
            # Save locally
            self.model.save_pretrained(self.model_dir)
            self.feature_extractor.save_pretrained(self.model_dir)
        else:
            print("Loading cached language detection model...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir, local_files_only=True)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_dir, local_files_only=True)
        
        self.model = self.model.to(self.device)
    
    def detect_language(self, audio_data):
        """
        Detect language from audio data.
        Returns tuple of (language_code, confidence, all_predictions)
        """
        try:
            # Prepare input features
            inputs = self.feature_extractor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(predictions, dim=-1).item()
            
            # Get confidence scores for all languages
            confidence_scores = predictions[0].cpu().numpy()
            all_predictions = [
                (self.id2label[i], float(confidence_scores[i]))
                for i in range(len(self.id2label))
            ]
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top prediction
            predicted_language = self.id2label[predicted_id]
            confidence = confidence_scores[predicted_id]
            
            # Get ISO language code
            language_code = self.get_iso_code(predicted_language)
            
            return language_code, confidence, all_predictions
            
        except Exception as e:
            print(f"Error in language detection: {e}")
            return "en", 0.0, []
    
    def get_iso_code(self, language_name):
        """Convert language name to ISO code."""
        language_codes = {
            "Arabic": "ar", "Czech": "cs", "German": "de", "English": "en",
            "Spanish": "es", "Persian": "fa", "French": "fr", "Hindi": "hi",
            "Italian": "it", "Japanese": "ja", "Korean": "ko", "Dutch": "nl",
            "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Turkish": "tr",
            "Vietnamese": "vi", "Chinese": "zh", "Telugu": "te", "Urdu": "ur"
        }
        return language_codes.get(language_name, "en")

def load_audio(file_path):
    """Load audio file."""
    audio, sr = librosa.load(file_path, sr=16000)
    return audio, sr

def transcribe(audio_data):
    """Wrapper function for compatibility."""
    transcriber = Wav2VecTranscriber()  # Your existing transcriber class
    lang_detector = LanguageDetector()
    
    # Get language with confidence scores
    language_code, confidence, all_predictions = lang_detector.detect_language(audio_data)
    
    # Print detailed language analysis
    print("\nLanguage Detection Results:")
    print(f"Primary Language: {[pred[0] for pred in all_predictions if pred[0] == language_code][0]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop 5 Language Predictions:")
    for lang, conf in all_predictions[:5]:
        print(f"{lang}: {conf:.2%}")
    
    # Get transcription
    transcription = transcriber.transcribe(audio_data)
    
    return transcription, language_code