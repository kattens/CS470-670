# Import necessary functions from whisperevaluation.py and bertanalysis.py
from evaluation.whisperevaluation import load_audio, transcribe_audio
from evaluation.bertanalysis import analyze_sentiment

def main():
    # Specify the path to your audio file
    audio_file_path = "temp_audio.wav"  # Replace with your actual audio file path

    # Load the audio data
    audio_data, sample_rate = load_audio(audio_file_path)

    # Transcribe the audio and get the detected language
    transcription, language = transcribe_audio(audio_data)

    # Print the transcription and detected language
    print("Transcription:", transcription)
    print("Detected Language:", language)

    # Analyze sentiment of the transcription
    sentiment_result = analyze_sentiment(transcription)

    # Print sentiment results
    print("Sentiment Analysis Result:", sentiment_result)

if __name__ == "__main__":
    main()
