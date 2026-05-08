"""
Sahl UFC Predictor - Backtest Engine
=====================================
Simulates historical betting performance: takes the trained model + Kaggle
dataset (which has historical odds) and computes what would've happened if
you'd followed Sahl's positive-EV picks across all past fights.

Outputs to data/backtest_results.json - the website reads this for
the "How Sahl performed historically" dashboard panel.
"""

from __future__ import annotations
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("backtest")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def american_to_decimal(odds):
    if pd.isna(odds): return np.nan
    if odds > 0: return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def kelly_fraction(p, decimal_odds, cap=0.5, bankroll_pct_cap=0.05):
    """Half-Kelly with 5% bankroll cap."""
    b = decimal_odds - 1.0
    if b <= 0 or pd.isna(b): return 0.0
    q = 1.0 - p
    f = max(0.0, (p * b - q) / b)
    return min(f * cap, bankroll_pct_cap)


def main():
    feat_csv = DATA / "features.csv"
    raw_csv = DATA / "ufc_fights_raw.csv"
    if not feat_csv.exists() or not raw_csv.exists():
        log.error("Missing features.csv or ufc_fights_raw.csv. Run features.py first.")
        sys.exit(1)

    feat = pd.read_csv(feat_csv)
    log.info("Loaded %d fights for backtest", len(feat))

    booster = xgb.Booster()
    booster.load_model(str(MODELS / "xgb_model.json"))
    with open(MODELS / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(MODELS / "feature_list.json") as f:
        feature_cols = json.load(f)

    # Read original Kaggle file for the odds columns we need
    kaggle_csv = DATA / "kaggle_raw" / "ufc-master.csv"
    if kaggle_csv.exists():
        k = pd.read_csv(kaggle_csv)
        odds_map = {}
        for _, r in k.iterrows():
            key = (str(r.get("R_fighter", "")), str(r.get("B_fighter", "")), str(r.get("date", "")))
            oa, ob = r.get("R_odds"), r.get("B_odds")
            if pd.notna(oa) and pd.notna(ob):
                odds_map[key] = (oa, ob)
        log.info("Loaded odds for %d historical fights", len(odds_map))
    else:
        odds_map = {}
        log.warning("No Kaggle odds file found. Backtest will be empty.")

    feat = feat[feat["is_decided"] == 1].sort_values("event_date").reset_index(drop=True)
    X = feat[feature_cols].values
    d = xgb.DMatrix(X, feature_names=feature_cols)
    raw_p = booster.predict(d)
    p_a = calibrator.transform(raw_p)
    p_a = np.clip(p_a, 0.01, 0.99)

    # Backtest only on the most recent 30% (not used in training)
    cut = int(len(feat) * 0.70)
    bt = feat.iloc[cut:].copy().reset_index(drop=True)
    bt["p_a"] = p_a[cut:]
    bt["p_b"] = 1 - bt["p_a"]

    bankroll = 1000.0
    starting_bankroll = bankroll
    history = [{"date": bt["event_date"].iloc[0] if len(bt) else "start",
                "bankroll": bankroll}]
    bets_placed = 0
    bets_won = 0
    total_staked = 0.0
    peak_bankroll = bankroll
    max_drawdown = 0.0
    yearly = {}
    skipped_no_odds = 0

    for _, fight in bt.iterrows():
        date = str(fight["event_date"])
        year = date[:4]
        a, b = fight["fighter_a"], fight["fighter_b"]
        won_a = fight["a_won"] == 1

        oa, ob = odds_map.get((str(a), str(b), date), (None, None))
        if oa is None or ob is None:
            skipped_no_odds += 1
            continue

        dec_a = american_to_decimal(oa)
        dec_b = american_to_decimal(ob)
        ev_a = fight["p_a"] * (dec_a - 1) - (1 - fight["p_a"])
        ev_b = fight["p_b"] * (dec_b - 1) - (1 - fight["p_b"])

        if ev_a > ev_b and ev_a > 0.02:
            f = kelly_fraction(fight["p_a"], dec_a)
            stake = f * bankroll
            if stake < 1: continue
            bets_placed += 1
            total_staked += stake
            profit = stake * (dec_a - 1) if won_a else -stake
            if won_a: bets_won += 1
            bankroll += profit
            yearly.setdefault(year, {"profit": 0, "bets": 0, "wins": 0})
            yearly[year]["profit"] += profit
            yearly[year]["bets"] += 1
            yearly[year]["wins"] += int(won_a)
        elif ev_b > ev_a and ev_b > 0.02:
            f = kelly_fraction(fight["p_b"], dec_b)
            stake = f * bankroll
            if stake < 1: continue
            bets_placed += 1
            total_staked += stake
            profit = stake * (dec_b - 1) if not won_a else -stake
            if not won_a: bets_won += 1
            bankroll += profit
            yearly.setdefault(year, {"profit": 0, "bets": 0, "wins": 0})
            yearly[year]["profit"] += profit
            yearly[year]["bets"] += 1
            yearly[year]["wins"] += int(not won_a)
        else:
            continue

        history.append({"date": date, "bankroll": round(bankroll, 2)})
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = (peak_bankroll - bankroll) / peak_bankroll
        if dd > max_drawdown:
            max_drawdown = dd

    final_pl = bankroll - starting_bankroll
    roi = final_pl / total_staked if total_staked > 0 else 0
    win_rate = bets_won / bets_placed if bets_placed > 0 else 0

    yearly_summary = {
        y: {
            "profit": round(s["profit"], 2),
            "bets": s["bets"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["bets"], 3) if s["bets"] else 0,
        }
        for y, s in sorted(yearly.items())
    }

    results = {
        "starting_bankroll": starting_bankroll,
        "final_bankroll": round(bankroll, 2),
        "total_pl": round(final_pl, 2),
        "roi_pct": round(roi * 100, 2),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "win_rate": round(win_rate, 3),
        "total_staked": round(total_staked, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "peak_bankroll": round(peak_bankroll, 2),
        "skipped_no_odds": skipped_no_odds,
        "history": history,
        "yearly": yearly_summary,
        "test_window": f"{bt['event_date'].min()} to {bt['event_date'].max()}" if len(bt) else "",
    }

    out_path = DATA / "backtest_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Backtest: %d bets, %.1f%% win rate, $%.2f P&L (%.2f%% ROI)",
             bets_placed, win_rate * 100, final_pl, roi * 100)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
