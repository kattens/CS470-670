from transformers import pipeline

def analyze_sentiment(text):
    # Load a sentiment-analysis pipeline with mBERT
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    # Analyze sentiment of the provided text
    sentiment_result = sentiment_pipeline(text)
    
    return sentiment_result