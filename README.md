1. **Common Voice Dataset**:
   - Input: Contains fields such as audio, age, text, and gender.
   - Output: This data is used as input to various models:
     - **Audio-to-Text Conversion**:
       - Model: **Whisper**
       - Input: Audio
       - Output: Text
     - **Audio-to-Age/Gender Classification**:
       - Model: **Wav2Vec2**
       - Input: Audio
       - Output: Age or Gender
   - **Text Processing**:
     - Text generated by Whisper is further processed to determine **language**.

2. **Sentiment Data**:
   - Input: Audio and sentiment labels.
   - Model: **Multilingual BERT**
   - Input: Audio + Sentiment labels
   - Output: Predicted sentiment.

### Summary
The workflow processes the Common Voice and sentiment datasets to perform:
- Speech-to-text using Whisper.
- Age/gender classification from audio using Wav2Vec2.
- Language determination from text.
- Sentiment analysis using Multilingual BERT.
