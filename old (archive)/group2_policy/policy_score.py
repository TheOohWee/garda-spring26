import os
import re
import pandas as pd
from transformers import pipeline

# --- CONFIGURATION ---
TOPIC_KEYWORDS = {
    "Inflation/Prices": ["inflation", "price", "cpi", "pce", "deflation", "sticky"],
    "Labor Market": ["job", "employment", "unemployment", "wage", "labor", "payroll"],
    "Interest Rates": ["rate", "hike", "cut", "basis point", "bps", "yield"],
    "Growth/GDP": ["growth", "gdp", "recession", "expansion", "economy", "activity"]
}

def extract_metadata(text: str):
    """Scans the top of the document to figure out the Date and the Central Bank."""
    # Expanded to the first 2500 characters to catch delayed introductions
    header = text[:2500]
    
    # 1. Extract the Source (Central Bank) using expanded keyword nets
    source = "Unknown Bank"
    
    # Fed: Look for FOMC, Powell, or the formal name
    if re.search(r'\b(Federal Reserve|FOMC|Board of Governors|Federal Open Market Committee|Jerome Powell|Chair Powell)\b', header, re.IGNORECASE):
        source = "US Federal Reserve"
        
    # ECB: Look for ECB, Lagarde, or Frankfurt
    elif re.search(r'\b(European Central Bank|ECB|Christine Lagarde|President Lagarde|Frankfurt)\b', header, re.IGNORECASE):
        source = "European Central Bank"
        
    # BoC: Look for BoC, Macklem, or Ottawa
    elif re.search(r'\b(Bank of Canada|BoC|Tiff Macklem|Governor Macklem)\b', header, re.IGNORECASE):
        source = "Bank of Canada"
        
    # BoE: Look for BoE, Bailey, or MPC
    elif re.search(r'\b(Bank of England|BoE|Andrew Bailey|Governor Bailey|Monetary Policy Committee|MPC)\b', header, re.IGNORECASE):
        source = "Bank of England"
        
    # BoJ: Look for BoJ, Ueda, Kuroda, or Policy Board
    elif re.search(r'\b(Bank of Japan|BoJ|Kazuo Ueda|Governor Ueda|Haruhiko Kuroda|Governor Kuroda|Policy Board)\b', header, re.IGNORECASE):
        source = "Bank of Japan"
        
    # 2. Extract the Date
    # Looks for standard formats like: "March 20, 2024", "20 March 2024", or "Oct. 31, 2023"
    date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December|[A-Z][a-z]{2}\.?)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
    date_match = re.search(date_pattern, header, re.IGNORECASE)
    
    if date_match:
        try:
            parsed_date = pd.to_datetime(date_match.group(0)).strftime('%Y-%m-%d')
        except:
            parsed_date = date_match.group(0).strip()
    else:
        parsed_date = "Unknown Date"
        
    return source, parsed_date

def identify_topic(sentence: str) -> str:
    """Scans the sentence for keywords to determine what it's talking about."""
    sentence_lower = sentence.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(word in sentence_lower for word in keywords):
            return topic
    return "General Macro"

def clean_and_split(text: str) -> list:
    """Cleans the text and splits it into individual sentences."""
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s for s in sentences if len(s) > 15]

def main():
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(script_dir, "raw_docs")
    outputs_dir = os.path.join(script_dir, "outputs")
    
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    files_to_process = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    if not files_to_process:
        print(f"⚠️ No .txt files found in: {docs_dir}")
        return

    # 2. Load FinBERT
    print("Loading FinBERT AI...")
    analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    print("Model loaded successfully!\n")

    all_results = []

    # 3. Process the documents
    for filename in files_to_process:
        filepath = os.path.join(docs_dir, filename)
        
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            
        # Magically pull the date and source from INSIDE the text!
        source_bank, file_date = extract_metadata(text)
            
        print(f"📄 Analyzing: {filename}")
        print(f"   ➔ Detected Source: {source_bank}")
        print(f"   ➔ Detected Date:   {file_date}")
        
        sentences = clean_and_split(text)
        
        for sentence in sentences:
            try:
                ai_result = analyzer(sentence)[0]
                label = ai_result["label"]
                
                # Positive = Hawkish (+1), Negative = Dovish (-1), Neutral = (0)
                score_val = 1 if label == "positive" else -1 if label == "negative" else 0
                
                all_results.append({
                    "Source_Bank": source_bank,
                    "Date": file_date,
                    "Topic": identify_topic(sentence),
                    "Sentence": sentence,
                    "Sentiment_Label": label,
                    "Numeric_Score": score_val,
                    "AI_Confidence": round(ai_result["score"], 4),
                    "Original_File": filename
                })
            except Exception as e:
                pass

    # 4. Generate the Outputs
    if all_results:
        df_sentences = pd.DataFrame(all_results)
        
        # Output 1: The Granular Data
        sentences_out = os.path.join(outputs_dir, "1_detailed_sentences.csv")
        df_sentences.to_csv(sentences_out, index=False)
        
        # Output 2: Macro Score Grouped by SOURCE and DATE
        df_macro = df_sentences.groupby(['Source_Bank', 'Date']).agg(
            Daily_Macro_Score=('Numeric_Score', 'mean'),
            Total_Sentences=('Sentence', 'count')
        ).reset_index()
        
        df_macro['Daily_Macro_Score'] = df_macro['Daily_Macro_Score'].round(3)
        
        macro_out = os.path.join(outputs_dir, "2_daily_macro_scores_by_bank.csv")
        df_macro.to_csv(macro_out, index=False)
        
        # Output 3: Topic Score Grouped by SOURCE and DATE
        df_topic_macro = df_sentences.groupby(['Source_Bank', 'Date', 'Topic']).agg(
            Topic_Score=('Numeric_Score', 'mean'),
            Sentence_Count=('Sentence', 'count')
        ).reset_index()
        
        df_topic_macro['Topic_Score'] = df_topic_macro['Topic_Score'].round(3)
        
        topic_out = os.path.join(outputs_dir, "3_daily_topic_scores_by_bank.csv")
        df_topic_macro.to_csv(topic_out, index=False)

        print("\n✅ Success! Analysis complete.")
        print("Generated files in your outputs folder:")
        print(f"  1. {sentences_out}")
        print(f"  2. {macro_out} (Overall scores separated by Fed vs. ECB vs. BoC)")
        print(f"  3. {topic_out} (Topic scores separated by Bank)")

if __name__ == "__main__":
    main()