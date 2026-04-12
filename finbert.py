import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import csv
import re
from pathlib import Path

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


def split_into_sentences(text):
    """Split text into sentences using common sentence delimiters."""
    # Split by common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def read_policy_transcripts(folder_path):
    """Read all txt files from policy_transcripts folder."""
    txt_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.txt'):
                txt_files.append(os.path.join(folder_path, file))
    return txt_files


# Define financial metrics with associated keywords
FINANCIAL_METRICS = {
    'demand': ['demand', 'consumption', 'spending', 'purchasing', 'buying', 'market demand', 'consumer demand'],
    'hiring': ['hiring', 'employment', 'jobs', 'labor', 'workforce', 'recruitment', 'staffing', 'unemployment'],
    'pricing': ['pricing', 'price', 'inflation', 'costs', 'rates', 'fees', 'tariffs', 'interest rates'],
    'capex': ['capex', 'capital expenditure', 'investment', 'infrastructure', 'construction', 'equipment', 'facilities'],
    'AI': ['artificial intelligence', 'AI', 'machine learning', 'automation', 'technology', 'digital', 'innovation'],
    'GDP': ['GDP', 'gross domestic product', 'economic growth', 'economy', 'output'],
    'housing': ['housing', 'real estate', 'property', 'mortgage', 'home prices', 'construction'],
    'trade': ['trade', 'imports', 'exports', 'tariffs', 'commerce', 'international trade'],
    'monetary_policy': ['monetary policy', 'interest rates', 'federal reserve', 'central bank', 'quantitative easing'],
    'fiscal_policy': ['fiscal policy', 'budget', 'deficit', 'taxes', 'spending', 'government spending']
}


def is_sentence_relevant_to_metric(sentence, metric_keywords):
    """Check if a sentence contains keywords related to a specific metric."""
    sentence_lower = sentence.lower()
    return any(keyword.lower() in sentence_lower for keyword in metric_keywords)


def analyze_transcripts_with_metric_filtering(folder_path, output_csv="finbert_metric_analysis.csv"):
    """Read txt files from policy_transcripts folder and apply FinBERT ratings filtered by financial metrics."""
    print(f"Reading txt files from {folder_path}...\n")
    print(f"Analyzing for metrics: {', '.join(FINANCIAL_METRICS.keys())}\n")
    
    # Load tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    txt_files = read_policy_transcripts(folder_path)
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return
    
    # Collect all results
    all_results = []
    
    for file_path in txt_files:
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences
        sentences = split_into_sentences(content)
        
        if sentences:
            # Analyze each sentence for relevance to metrics
            relevant_sentences = []
            sentence_metrics = []  # Track which metrics each sentence relates to
            
            for sentence in sentences:
                sentence_relevant_metrics = []
                for metric, keywords in FINANCIAL_METRICS.items():
                    if is_sentence_relevant_to_metric(sentence, keywords):
                        sentence_relevant_metrics.append(metric)
                
                if sentence_relevant_metrics:  # Only analyze if relevant to at least one metric
                    relevant_sentences.append(sentence)
                    sentence_metrics.append(sentence_relevant_metrics)
            
            if relevant_sentences:
                # Get sentiment analysis results for relevant sentences
                results = finbert_pipeline(relevant_sentences)
                
                for sentence, result, metrics in zip(relevant_sentences, results, sentence_metrics):
                    # Create a result entry for each metric this sentence relates to
                    for metric in metrics:
                        all_results.append({
                            'filename': os.path.basename(file_path),
                            'sentence': sentence,
                            'metric': metric,
                            'sentiment': result['label'],
                            'confidence': result['score']
                        })
        
        print(f"  Found {len(relevant_sentences) if 'relevant_sentences' in locals() else 0} relevant sentences\n")
    
    # Save results to CSV
    if all_results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'sentence', 'metric', 'sentiment', 'confidence'])
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to {output_csv}")
        print(f"Total metric-sentence pairs analyzed: {len(all_results)}")
        
        # Print summary by metric
        metric_summary = {}
        sentiment_summary = {}
        
        for result in all_results:
            metric = result['metric']
            sentiment = result['sentiment']
            
            if metric not in metric_summary:
                metric_summary[metric] = 0
            metric_summary[metric] += 1
            
            key = f"{metric}_{sentiment}"
            if key not in sentiment_summary:
                sentiment_summary[key] = 0
            sentiment_summary[key] += 1
        
        print("\nAnalysis Summary by Metric:")
        for metric in sorted(FINANCIAL_METRICS.keys()):
            if metric in metric_summary:
                count = metric_summary[metric]
                print(f"\n  {metric.upper()}: {count} sentences")
                
                # Show sentiment breakdown for this metric
                for sentiment in ['positive', 'negative', 'neutral']:
                    key = f"{metric}_{sentiment}"
                    if key in sentiment_summary:
                        print(f"    {sentiment.capitalize()}: {sentiment_summary[key]}")
            else:
                print(f"\n  {metric.upper()}: 0 sentences")


if __name__ == "__main__":
    # Analyze all txt files in policy_transcripts folder with metric filtering
    transcripts_folder = "policy_transcripts"
    analyze_transcripts_with_metric_filtering(transcripts_folder)