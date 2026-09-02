# Prediction pipeline

Ingest historical fixtures → build leak-free features → train calibrated models
→ backtest → serve predictions to `predictor-web`.

## One-time setup

```bash
export DATABASE_URL=...          # your Postgres URL
export FOOTBALL_API_KEY=...      # api-football key (rotate the leaked one first)
alembic upgrade head             # adds results/odds columns to matches
```

## 1. Ingest history

```bash
python scripts/ingest_history.py --leagues 39,140,135,78,61 --seasons 2021,2022,2023,2024
```

League ids are api-football's: 39 Premier League, 140 La Liga, 135 Serie A,
78 Bundesliga, 61 Ligue 1.

One request per league-season. `--with-stats` and `--with-odds` add one request
*per fixture*, so they are slow and quota-hungry — but odds are required for the
value simulation, which is the only measure of whether predictions are worth
anything. Budget for them.

Aim for **at least 3 seasons across several leagues** (~4,000+ matches). Training
refuses to run below 500 labelled rows.

## 2. Train

```bash
python scripts/train_model.py
```

Writes `match_winner`, `over_under_2_5`, `btts` models plus `training_report.json`
to `$MODEL_DIR`. Each target reports `beats_baseline` — if that is `false`, the
model has learned nothing beyond base rates and must not be shipped.

## 3. Backtest

```bash
python scripts/backtest.py --folds 5 --edge 0.05
```

Reports per-fold accuracy and log loss, a calibration table, and a flat-stake ROI
simulation over selections where the model beats the market by `--edge`.

## 4. Serve

```bash
uvicorn app.main:app --port 8000                       # backend
cd ../bet_prediction_project/artifacts/predictor-web
pnpm dev                                               # proxies /api to :8000
```

## Design notes

**Why the split is temporal, never random.** Football is a time series. A random
train/test split lets the model learn from matches played *after* the ones it is
scored on. That inflates measured accuracy and collapses on real fixtures. The
previous `train.py` used `train_test_split(..., random_state=42)`; the current one
sorts by kickoff and holds out the most recent 20%.

**Why post-match stats are not features.** Possession, shots on target and xG are
only known *after* a match. Feeding a match's own stats into its prediction is
target leakage and produces a model that looks excellent offline and is useless
in production. Here they enter only as a team's rolling average over *previous*
fixtures. `tests/test_features.py` enforces this: rewriting a later result must
not change any earlier row's features.

**Why log loss, not accuracy.** The product needs trustworthy probabilities. A
model that is 55% accurate but wrong about its own confidence is worse than one
slightly less accurate and well calibrated. Models are selected on log loss and
wrapped in `CalibratedClassifierCV`, and `calibration_table()` checks that a
stated 70% actually lands about 70% of the time.

## What accuracy to expect

Roughly **50–55%** on 1X2 across a full season. Published academic models reach
about 53%; well-funded betting syndicates operate near 55–60%. If a run reports
much above 60%, assume leakage and go looking for it rather than celebrating.

Accuracy is also not the goal — beating the closing odds is. The market already
prices in public information, so the number that matters is the ROI from
`value_simulation()`. A model can be 55% accurate and still lose money, and a
selective one can be 45% accurate and profit. Expect early runs to show negative
ROI; that is the normal starting point, not a bug.
