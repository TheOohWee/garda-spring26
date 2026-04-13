======================================
FINBERT PIPELINE INSTRUCTIONS
======================================

1. PREP YOUR FILES
Place all raw central bank transcripts (must be .txt files) into the `raw_docs` folder.

2. INSTALL DEPENDENCIES
Ensure you have Python 3 installed. Open your terminal and install the required libraries by running:
pip install pandas transformers torch

3. RUN THE SCRIPT
Navigate to the project folder in your terminal and execute the pipeline by running:
python3 policy_score.py

4. VIEW YOUR DATA
Once finished, check the auto-generated `outputs` folder for your results:
- 1_detailed_sentences.csv (Every sentence mapped to a topic and sentiment label)
- 2_daily_macro_scores_by_bank.csv (Overall daily Hawkish/Dovish score per bank)
- 3_daily_topic_scores_by_bank.csv (Daily scores broken down by specific economic topics)