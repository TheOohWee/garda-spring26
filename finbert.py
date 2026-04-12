import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyze_sentiment_with_pipeline(texts):
    print("--- Using Pipeline Method ---")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    results = finbert_pipeline(texts)
    
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Confidence: {result['score']:.4f})\n")


if __name__ == "__main__":
    financial_sentences = [
        "The company reported a massive surge in profits for the third quarter.",
        "Due to severe supply chain constraints, our revenue plummeted by 20%.",
        "The board of directors is scheduling a standard meeting next Tuesday to discuss the ongoing projects."
    ]
    
    analyze_sentiment_with_pipeline(financial_sentences)