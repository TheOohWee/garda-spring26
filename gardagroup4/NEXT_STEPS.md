# Next Steps

I looked through the repo and wrote this up mostly as a path from where we are now to something we can actually test over time.

There is already a solid base here:

- macro, policy, and earnings/AI are all at least started
- there is already code that scores text and merges outputs
- the repo already makes tables/charts, which is useful for the final presentation

What we need now is the layer that turns this into a real strategy project instead of just a one-time snapshot.

## What the project should become

The end goal should be pretty simple:

1. build dated regional signals from macro data, central bank commentary, and earnings commentary
2. map those signals into actual trades in the products we care about
3. backtest those trades over time
4. show what worked, what did not, and why

Products in scope:

- equities: US, Europe, Japan
- G5 10y yields: US, Europe, UK, Japan, Canada
- gold
- oil
- CDX
- dollar

## Where the repo is right now

My read is that we already have the scoring side started, but not the full time-series / strategy side yet.

What is already there:

- Group 2 can score central bank text
- Group 3 can score earnings transcripts and AI/labor language
- `main_model.py` can merge outputs and create rankings/charts

What we still need:

- historical signal outputs by date
- market return data for the target assets
- rules that turn signals into trades
- a backtest engine

So the main shift now is: stop thinking only in terms of one final combined score, and start thinking in terms of signal history plus trade history.

## The main thing to build around

Every signal needs a date.

That is probably the most important rule for the rest of the project.

For every output row, we should know:

- what date the score is for
- when that information would have been knowable
- what region it belongs to
- what pillar it came from
- how much data went into it

If we do that part well, the backtest becomes much easier later.

## What each group should be producing

### Group 1: macro

This should end up as a monthly regional macro score history.

At minimum it should cover:

- growth
- inflation
- labor

Nice to have if there is time:

- confidence
- industrial activity
- housing / credit-sensitive stuff

Output we want:

- monthly macro score by region
- ideally with underlying sub-scores too

### Group 2: policy

This should be treated as an event-based signal.

For each doc or event we should know:

- bank
- date
- doc type
- region
- topic scores
- aggregate score

Then that should roll into a daily or monthly policy score by region.

Output we want:

- event-level scored policy dataset
- daily/monthly policy score by region

### Group 3: earnings + AI/labor

This should become a bottom-up economic panel over time.

The important thing here is not just scoring one transcript per company. It should become:

- company
- quarter / call date
- region
- sector
- theme scores

Themes that seem most useful:

- demand
- pricing power
- hiring/headcount
- capex
- credit conditions
- AI adoption
- productivity / labor displacement

Output we want:

- company-quarter scores
- regional earnings score over time
- regional AI/labor score over time

## What the combined model should do

Once Groups 1, 2, and 3 are all producing dated outputs, the combined model should:

1. standardize scores onto a comparable scale
2. keep track of coverage and missingness
3. combine the pillars into a dated composite signal
4. show which pillars are driving each regional score

So really the combined model should answer:

- what did we think about each region on a given date
- why did we think that
- what trade would that have pointed to

## Recommended build order

This is the order I would do things in.

### 1. Create one shared output format

All groups should export data in a similar structure.

Suggested columns:

- `as_of_date`
- `effective_date`
- `region`
- `pillar`
- `subpillar`
- `score_raw`
- `score_std`
- `coverage`
- `source_count`

This is a small thing but it will make everything else easier.

### 2. Make Group 1 a real pipeline

Instead of static files getting hand-read in integration, Group 1 should produce a clean macro score history.

To do:

- create a proper `macro_score.py`
- define what metrics go in
- define signs / transformations
- export monthly scores by region

Goal:

- be able to plot macro score history by region

### 3. Make Group 2 fully date-aware

The policy work is already useful, it just needs to be preserved in a more backtest-friendly format.

To do:

- keep scores by event date
- keep doc metadata
- roll events into daily/monthly regional policy scores
- give different weight to different doc types if that makes sense

Goal:

- be able to ask what the policy score was for a region at a specific point in time

### 4. Expand Group 3 into a historical panel

This is probably the part with the most upside, especially for the bottom-up / Druckenmiller angle.

To do:

- collect multiple quarters of transcripts per company
- add transcript dates and quarter tags
- balance region coverage better
- expand company selection intentionally by sector
- keep the earnings and AI/labor outputs separate

Goal:

- be able to see how company commentary changed over time and how that rolled into regional trends

### 5. Build the AI/labor piece out more clearly

Since that was part of the prompt, it should be its own clear sub-theme and not just an extra column.

Things worth tracking:

- AI/productivity language
- hiring slowdown
- layoffs / rightsizing
- staffing/recruiting company commentary
- young labor market data if we can get it

Goal:

- have a time-series signal that says something specific about AI and labor, not just "AI was mentioned a lot"

### 6. Rebuild the integration layer

Once the group outputs are more solid, then rebuild the composite model around them.

To do:

- combine the pillars by date and region
- standardize them consistently
- track missing data and stale data
- avoid over-forcing rankings when coverage is thin
- keep pillar contribution outputs

Goal:

- for any date, explain the regional ranking and what drove it

### 7. Build the asset mapping layer

This is where the signals become trades.

We need to define how regional and thematic views map into:

- cross-region equity trades
- duration / yield views
- dollar views
- gold / oil / CDX views

Goal:

- every signal should have a clear tradable expression

### 8. Build the backtest

This is the biggest missing piece right now.

I would start simple:

- monthly rebalancing
- realistic signal lag
- transaction cost assumption
- simple position sizing

Outputs we want:

- cumulative return
- Sharpe
- drawdown
- hit rate
- turnover
- contribution by asset or pillar

Goal:

- one honest end-to-end backtest that we can explain

### 9. Tune and improve after the backtest exists

After the historical setup is working, then we can spend time on:

- improving scoring methods
- changing weights
- trying better text models
- testing different portfolio rules

That way we are improving against something measurable instead of just guessing.

## Good first sprint

If we want a realistic next sprint, I would do this:

1. create one shared dated schema for outputs
2. fix the current fragile loaders in the integration layer
3. build a return table for the target assets
4. preserve policy dates through the full pipeline
5. add transcript quarter/date metadata to Group 3
6. build a simple monthly backtest skeleton

If we get through those six things, the project will be in much better shape.

## Suggested work split

One possible way to divide it up:

### Person 1

Data structure / reproducibility

- folder cleanup
- shared schemas
- processed outputs
- root dependencies
- tests

### Person 2

Macro

- clean macro inputs
- define transformations
- build monthly macro scores

### Person 3

Policy

- document metadata
- event scoring
- topic scoring
- daily/monthly rollups

### Person 4

Earnings + AI/labor

- transcript collection
- company metadata
- theme scoring
- historical aggregation

### Person 5

Integration + strategy

- composite scores
- asset mapping
- backtest engine
- strategy evaluation

## Stuff not to over-focus on yet

- making the charts prettier
- over-tuning model weights
- building a fancy dashboard
- wrapping everything in "AI agent" language before the research logic is there

Those things can come later. The main value right now is getting a clean historical pipeline in place.

## What a strong final version probably looks like

For the class, I think a strong final version would be:

- a few years of signal history
- regional macro, policy, and earnings/AI-labor scores
- a clear mapping from scores to trades
- one clean backtest with realistic lags
- charts showing both signals and performance
- a short writeup on what worked, what did not, and what we would improve next

That would already be a strong project.

## Bottom line

I think the base is there. The next step is just to organize it around time-series signals and an actual backtest.

If we can answer these three questions, we are moving in the right direction:

1. what was the score on a past date
2. what trade would that have implied
3. how would that trade have done after realistic lags and costs

That is probably the cleanest way to think about the whole build from here.
