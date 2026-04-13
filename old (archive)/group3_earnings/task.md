Group 1: Macro + Data Infrastructure
Group 2: Central Bank NLP
Group 3: Earnings + AI Labor Signals
Group 4: Integration + Portfolio Output

Group 1: Macro + Data Infrastructure (Kellan + Nicholas)
Pull macro data from FRED/BLS
Focus on growth, inflation, and labor data
Clean and organize data by region/country
Build macro scoring engine
Compute level z-scores
Compute surprise z-scores
Combine into one macro score
Export final macro score for each region
Deliverables:
clean macro dataset
macro_score.py
output file with regional macro scores


Group 2: Central Bank NLP (Ben + Leo)
Collect central bank statements, speeches, and press conference text
Focus first on Fed, ECB, BOE, BOJ, BOC if possible
Build simple scoring system for:
inflation
labor
growth
forward guidance
Assign hawkish/dovish sentiment
Assign magnitude score
Apply tier weighting
Sum scores into one policy score by region
Deliverables:
cleaned text dataset
policy_score.py
sample scored excerpts
output file with regional policy scores


Group 3: Earnings + AI Labor Signals (Aakaash + Amir)
Pull a small set of earnings call transcripts from representative companies
Start with about 10–20 companies total
Track themes like:
demand
hiring
pricing
capex
AI
efficiency
Build simple keyword/sentiment scoring
Create earnings signal by region
Build AI labor/productivity proxy score using a few simple indicators
Deliverables:
earnings_score.py
ai_score.py
transcript/theme summary file
output file with regional earnings and AI scores


Group 4: Integration + Portfolio Output (Magdalena + Kris)
Take outputs from Groups 1–3
Standardize all scores to one comparable scale
Combine into one composite score
Rank regions from strongest to weakest
Build simple trade mapping logic
Produce 2–3 example trade ideas
Create charts and final output table
Make results presentation-ready
Deliverables:
main_model.py
composite score table
regional rankings
trade ideas
charts/visuals for deck
