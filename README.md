# Sahl UFC Predictor

> Calibrated XGBoost on 24 engineered features. Produces win probabilities, expected value vs. the book, and Half-Kelly stake recommendations.

## What it does

Given two fighters and (optionally) the market odds:

- Computes a **calibrated win probability** using a gradient-boosted model
- Compares against the book's **vig-adjusted implied probability** to find an edge
- Calculates **Expected Value** per dollar staked
- Recommends a **Half-Kelly stake**, capped at 5% of bankroll
- Explains each pick with **per-prediction SHAP contributions**
- Ranks a full UFC card by EV, surfacing the best bets first

## Architecture

```
sahl-ufc/
├── backend/
│   ├── scraper.py     # Polite ufcstats.com scraper (1 req/sec, retries)
│   ├── features.py    # Elo + 24 leakage-free features + synthetic generator
│   ├── train.py       # XGBoost + walk-forward CV + isotonic calibration + SHAP
│   ├── server.py      # FastAPI: /predict, /card, /metrics, /fighters
│   └── test_inproc.py # Smoke test (no HTTP)
├── frontend/
│   └── index.html     # Single-file SPA (fights, picks, SHAP bars, card mode)
├── data/              # CSVs land here
├── models/            # Trained XGB + calibrator + Elo + fighter state
├── output/            # reliability.png, shap_summary.png, metrics.json
└── requirements.txt
```

## The 24 Features

All differentials (A − B), all computed using **only data strictly before the fight date** (no leakage):

| # | Feature | Family |
|---|---|---|
| 1 | `elo_diff` | Skill rating |
| 2 | `elo_trend_diff` | 30-day Elo momentum |
| 3 | `win_streak_diff` | Current streak |
| 4 | `win_pct_diff` | Career UFC win % |
| 5 | `recent_form_diff` | Last-5 win % |
| 6 | `slpm_diff` | Sig. strikes landed/min |
| 7 | `sapm_diff` | Sig. strikes absorbed/min (inverted) |
| 8 | `str_acc_diff` | Striking accuracy |
| 9 | `str_def_diff` | Striking defense |
| 10 | `kd_per15_diff` | Knockdowns per 15 min |
| 11 | `td_per15_diff` | Takedowns landed per 15 min |
| 12 | `td_acc_diff` | Takedown accuracy |
| 13 | `td_def_diff` | Takedown defense |
| 14 | `ctrl_pct_diff` | Control time / total |
| 15 | `sub_att_per15_diff` | Submission attempts/15 |
| 16 | `getup_speed_diff` | Reversals/min (proxy) |
| 17 | `age_diff` | Age (years) |
| 18 | `height_diff_cm` | Height |
| 19 | `reach_diff_cm` | Reach |
| 20 | `weight_class_encoded` | Weight class ordinal |
| 21 | `stance_match` | Orthodox vs Southpaw flag |
| 22 | `layoff_diff_days` | Days since last fight |
| 23 | `ufc_exp_diff` | Prior UFC fights |
| 24 | `title_bout` | Title fight flag |

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Try the synthetic pipeline (works immediately, no scraping)
cd backend
python features.py --synthetic
python train.py
python server.py
# → open http://localhost:8000
```

## Training on real UFC data

```bash
# 1. Scrape ufcstats.com (10 years, ~1 hour, polite 1 req/sec)
cd backend
python scraper.py --output ../data/ufc_fights_raw.csv --years 10

# 2. (Optional) Augment with the Kaggle dataset
#    https://www.kaggle.com/datasets/rajeevw/ufcdata
#    https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset
#    Manually merge into ../data/ufc_fights_raw.csv keeping the same column schema.

# 3. Build features + train
python features.py
python train.py

# 4. Serve
python server.py
```

## Why calibration > raw accuracy

For betting, a model that says "70%" must actually win 70% of the time across all "70%" predictions. We use **isotonic regression** fit on a held-out validation set to map raw XGBoost outputs to calibrated probabilities. The reliability curve (saved to `output/reliability.png`) compares pre- and post-calibration. Brier score and log-loss matter more than accuracy here.

## Why walk-forward CV (not random K-fold)

UFC fights are time-series. Random shuffling lets the model "see" future fights when training to predict past ones — that's leakage. We grow the training set chronologically and validate on the next time slice, repeated 5×.

## Kelly criterion notes

- **Full Kelly** maximizes long-run growth but has crushing drawdowns.
- We default to **Half Kelly** (`kelly_fraction_cap=0.5`) and cap any single bet at **5% of bankroll** (`bankroll_cap_pct=0.05`).
- We only recommend a bet when EV > 2%.
- Swap to Quarter Kelly by setting `kelly_fraction_cap=0.25` in the request body.

## API

```
GET  /api/health        → {"status": "ok", "fighters_known": N}
GET  /api/fighters      → [{name, elo, fights, wins, losses, streak}, ...]
GET  /api/metrics       → {calibrated: {acc, logloss, brier, auc}, cv_folds: [...]}
POST /api/predict       → predict one fight (see PredictionOut schema)
POST /api/card          → predict a card, ranked by EV
```

### Example

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fighter_a": "Islam Makhachev",
    "fighter_b": "Charles Oliveira",
    "weight_class": "Lightweight",
    "title_bout": 1,
    "odds_a": -180,
    "odds_b": +160,
    "bankroll": 1000
  }'
```

## Realistic expectations

State-of-the-art UFC prediction tops out around **65-79% accuracy** on time-aware splits. MMA is genuinely high-variance — injuries, weight cuts, judging — so the goal isn't "predict every fight right." The edge for profitable betting comes from:

1. **Calibration**: knowing when 60% really means 60%
2. **Disagreement with the book**: spots where your probability and the implied probability diverge enough to overcome vig
3. **Discipline**: only betting positive-EV spots and sizing with Kelly

## License

MIT.
