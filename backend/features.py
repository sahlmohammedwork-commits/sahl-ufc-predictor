"""
Sahl UFC Predictor - Feature Engineering
==========================================
Builds 24 engineered features per fight using ONLY pre-fight info
(no leakage from the fight being predicted).

The 24 features (all are A_minus_B differentials unless noted):

  Skill / rating
   1.  elo_diff                  Elo rating differential
   2.  elo_diff_30d              30-day Elo trend differential
   3.  win_streak_diff           current win streak
   4.  win_pct_diff              career win % in UFC
   5.  recent_form_diff          win% over last 5 UFC fights

  Striking composite
   6.  slpm_diff                 sig. strikes landed / min
   7.  sapm_diff                 sig. strikes absorbed / min
   8.  str_acc_diff              sig. strike accuracy
   9.  str_def_diff              sig. strike defense (1 - accuracy_against)
  10.  kd_per15_diff             knockdowns per 15 min

  Grappling composite
  11.  td_per15_diff             takedowns landed per 15 min
  12.  td_acc_diff               takedown accuracy
  13.  td_def_diff               takedown defense
  14.  ctrl_pct_diff             control time / total fight time
  15.  sub_att_per15_diff        submission attempts per 15 min
  16.  getup_speed_diff          (proxy) reversals + escapes per minute on bottom

  Physical / context
  17.  age_diff                  fighter A age - fighter B age (years)
  18.  height_diff_cm
  19.  reach_diff_cm
  20.  weight_class_encoded      ordinal lightweight..heavyweight
  21.  stance_match              orthodox-vs-southpaw =1 else 0
  22.  layoff_diff_days          days since last fight (rest)
  23.  ufc_exp_diff              # of prior UFC fights
  24.  title_bout                1 if title fight (context only)

Time-aware: every feature is computed using ONLY data strictly before
the fight's date.

NOTE ON SYNTHETIC MODE:
  If --synthetic is passed, generates a realistic synthetic dataset so
  the pipeline can be demonstrated end-to-end without scraping. Replace
  with real CSV from scraper.py for production.
"""

from __future__ import annotations
import argparse
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("features")


WEIGHT_CLASS_ORDER = {
    "Strawweight": 1, "Flyweight": 2, "Bantamweight": 3, "Featherweight": 4,
    "Lightweight": 5, "Welterweight": 6, "Middleweight": 7,
    "Light Heavyweight": 8, "Heavyweight": 9,
    "Women's Strawweight": 1, "Women's Flyweight": 2,
    "Women's Bantamweight": 3, "Women's Featherweight": 4,
}


# ---------- Elo ----------

class EloRater:
    """Standard chess-style Elo, K=32, draws -> 0.5/0.5."""

    def __init__(self, k: float = 32.0, base: float = 1500.0):
        self.k = k
        self.base = base
        self.ratings: dict[str, float] = defaultdict(lambda: base)
        self.history: dict[str, list[tuple[str, float]]] = defaultdict(list)
        # history[fighter] = [(date, rating_after), ...]

    def get(self, fighter: str) -> float:
        return self.ratings[fighter]

    def get_at(self, fighter: str, date: str) -> float:
        """Rating *as of* the start of `date`."""
        hist = self.history.get(fighter, [])
        last = self.base
        for d, r in hist:
            if d < date:
                last = r
            else:
                break
        return last

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, winner: str, date: str):
        ra, rb = self.ratings[a], self.ratings[b]
        ea = self.expected(ra, rb)
        if winner == "a":   sa = 1.0
        elif winner == "b": sa = 0.0
        else:               sa = 0.5  # draw / NC
        sb = 1.0 - sa
        self.ratings[a] = ra + self.k * (sa - ea)
        self.ratings[b] = rb + self.k * (sb - (1 - ea))
        self.history[a].append((date, self.ratings[a]))
        self.history[b].append((date, self.ratings[b]))


# ---------- Rolling fighter stats ----------

@dataclass
class FighterRunning:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    streak: int = 0  # negative = losing
    last_5: list[int] = field(default_factory=list)  # 1=win,0=loss
    last_fight_date: Optional[str] = None
    # totals across all fights (career UFC)
    sec_in_cage: int = 0
    sig_str_lnd: int = 0
    sig_str_att: int = 0
    sig_str_lnd_against: int = 0
    sig_str_att_against: int = 0
    kd: int = 0
    td_lnd: int = 0
    td_att: int = 0
    td_lnd_against: int = 0
    td_att_against: int = 0
    sub_att: int = 0
    ctrl_sec: int = 0
    rev: int = 0


def fight_duration_seconds(round_no: int, time_str: str) -> int:
    """Approximate fight length in seconds (round * 5min minus remaining)."""
    if not round_no:
        return 5 * 60
    try:
        m, s = (time_str or "0:00").split(":")
        last = int(m) * 60 + int(s)
    except Exception:
        last = 5 * 60
    return (round_no - 1) * 5 * 60 + last


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


# ---------- Build features ----------

def build_features(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    fights_df columns expected (output of scraper):
      event_date, fighter_a, fighter_b, winner, weight_class, title_bout,
      method, round, time,
      a_kd, b_kd, a_sig_str_lnd, a_sig_str_att, b_sig_str_lnd, b_sig_str_att,
      a_td_lnd, a_td_att, b_td_lnd, b_td_att, a_sub_att, b_sub_att,
      a_rev, b_rev, a_ctrl_sec, b_ctrl_sec.
    Optional fighter physicals (joined later if available):
      a_age, b_age, a_height_cm, b_height_cm, a_reach_cm, b_reach_cm,
      a_stance, b_stance.
    """
    df = fights_df.sort_values("event_date").reset_index(drop=True)
    elo = EloRater()
    state: dict[str, FighterRunning] = defaultdict(FighterRunning)
    rows = []

    for _, f in df.iterrows():
        a, b = f["fighter_a"], f["fighter_b"]
        date = f["event_date"]
        sa, sb = state[a], state[b]

        # --- pre-fight features ---
        elo_a, elo_b = elo.get(a), elo.get(b)
        elo_a_30d_ago = elo.get_at(a, (datetime.fromisoformat(date) - timedelta(days=30)).date().isoformat()) if isinstance(date, str) else elo_a
        elo_b_30d_ago = elo.get_at(b, (datetime.fromisoformat(date) - timedelta(days=30)).date().isoformat()) if isinstance(date, str) else elo_b
        elo_trend_a = elo_a - elo_a_30d_ago
        elo_trend_b = elo_b - elo_b_30d_ago

        win_pct_a = safe_div(sa.wins, sa.fights, 0.5)
        win_pct_b = safe_div(sb.wins, sb.fights, 0.5)
        recent_a = safe_div(sum(sa.last_5), len(sa.last_5), 0.5) if sa.last_5 else 0.5
        recent_b = safe_div(sum(sb.last_5), len(sb.last_5), 0.5) if sb.last_5 else 0.5

        a_min = sa.sec_in_cage / 60.0
        b_min = sb.sec_in_cage / 60.0
        slpm_a = safe_div(sa.sig_str_lnd, a_min)
        slpm_b = safe_div(sb.sig_str_lnd, b_min)
        sapm_a = safe_div(sa.sig_str_lnd_against, a_min)
        sapm_b = safe_div(sb.sig_str_lnd_against, b_min)
        str_acc_a = safe_div(sa.sig_str_lnd, sa.sig_str_att, 0.4)
        str_acc_b = safe_div(sb.sig_str_lnd, sb.sig_str_att, 0.4)
        str_def_a = 1.0 - safe_div(sa.sig_str_lnd_against, sa.sig_str_att_against, 0.6)
        str_def_b = 1.0 - safe_div(sb.sig_str_lnd_against, sb.sig_str_att_against, 0.6)
        kd15_a = safe_div(sa.kd, a_min) * 15.0
        kd15_b = safe_div(sb.kd, b_min) * 15.0

        td15_a = safe_div(sa.td_lnd, a_min) * 15.0
        td15_b = safe_div(sb.td_lnd, b_min) * 15.0
        td_acc_a = safe_div(sa.td_lnd, sa.td_att, 0.3)
        td_acc_b = safe_div(sb.td_lnd, sb.td_att, 0.3)
        td_def_a = 1.0 - safe_div(sa.td_lnd_against, sa.td_att_against, 0.7)
        td_def_b = 1.0 - safe_div(sb.td_lnd_against, sb.td_att_against, 0.7)
        ctrl_pct_a = safe_div(sa.ctrl_sec, sa.sec_in_cage)
        ctrl_pct_b = safe_div(sb.ctrl_sec, sb.sec_in_cage)
        sub15_a = safe_div(sa.sub_att, a_min) * 15.0
        sub15_b = safe_div(sb.sub_att, b_min) * 15.0
        # get-up speed proxy: reversals per minute on bottom (we have rev count
        # but not bottom-time, so use rev / total min as proxy)
        getup_a = safe_div(sa.rev, a_min)
        getup_b = safe_div(sb.rev, b_min)

        a_age = f.get("a_age", np.nan)
        b_age = f.get("b_age", np.nan)
        a_h = f.get("a_height_cm", np.nan)
        b_h = f.get("b_height_cm", np.nan)
        a_r = f.get("a_reach_cm", np.nan)
        b_r = f.get("b_reach_cm", np.nan)
        a_stance = str(f.get("a_stance", "Orthodox")).lower()
        b_stance = str(f.get("b_stance", "Orthodox")).lower()
        stance_match = int(("orthodox" in a_stance) ^ ("orthodox" in b_stance))

        layoff_a = (datetime.fromisoformat(date) - datetime.fromisoformat(sa.last_fight_date)).days if sa.last_fight_date else 365
        layoff_b = (datetime.fromisoformat(date) - datetime.fromisoformat(sb.last_fight_date)).days if sb.last_fight_date else 365

        wc_enc = WEIGHT_CLASS_ORDER.get(str(f.get("weight_class", "")).strip(), 5)

        rows.append({
            "event_date": date,
            "fighter_a": a,
            "fighter_b": b,
            # 24 features
            "elo_diff": elo_a - elo_b,
            "elo_trend_diff": elo_trend_a - elo_trend_b,
            "win_streak_diff": sa.streak - sb.streak,
            "win_pct_diff": win_pct_a - win_pct_b,
            "recent_form_diff": recent_a - recent_b,
            "slpm_diff": slpm_a - slpm_b,
            "sapm_diff": sapm_b - sapm_a,    # lower absorbed = better
            "str_acc_diff": str_acc_a - str_acc_b,
            "str_def_diff": str_def_a - str_def_b,
            "kd_per15_diff": kd15_a - kd15_b,
            "td_per15_diff": td15_a - td15_b,
            "td_acc_diff": td_acc_a - td_acc_b,
            "td_def_diff": td_def_a - td_def_b,
            "ctrl_pct_diff": ctrl_pct_a - ctrl_pct_b,
            "sub_att_per15_diff": sub15_a - sub15_b,
            "getup_speed_diff": getup_a - getup_b,
            "age_diff": (a_age - b_age) if pd.notna(a_age) and pd.notna(b_age) else 0.0,
            "height_diff_cm": (a_h - b_h) if pd.notna(a_h) and pd.notna(b_h) else 0.0,
            "reach_diff_cm": (a_r - b_r) if pd.notna(a_r) and pd.notna(b_r) else 0.0,
            "weight_class_encoded": wc_enc,
            "stance_match": stance_match,
            "layoff_diff_days": layoff_a - layoff_b,
            "ufc_exp_diff": sa.fights - sb.fights,
            "title_bout": int(f.get("title_bout", 0) or 0),
            # target
            "a_won": 1 if f["winner"] == "a" else 0,
            "is_decided": 1 if f["winner"] in ("a", "b") else 0,
        })

        # --- update state with this fight (post-fight) ---
        dur = fight_duration_seconds(int(f.get("round", 0) or 0), str(f.get("time", "0:00")))
        for fighter, side, opp_side in [(a, "a", "b"), (b, "b", "a")]:
            s = state[fighter]
            s.fights += 1
            won = (f["winner"] == side)
            if f["winner"] in ("a", "b"):
                if won:
                    s.wins += 1
                    s.streak = max(1, s.streak + 1) if s.streak >= 0 else 1
                    s.last_5.append(1)
                else:
                    s.losses += 1
                    s.streak = min(-1, s.streak - 1) if s.streak <= 0 else -1
                    s.last_5.append(0)
                s.last_5 = s.last_5[-5:]
            s.last_fight_date = date
            s.sec_in_cage += dur
            s.sig_str_lnd += int(f.get(f"{side}_sig_str_lnd", 0) or 0)
            s.sig_str_att += int(f.get(f"{side}_sig_str_att", 0) or 0)
            s.sig_str_lnd_against += int(f.get(f"{opp_side}_sig_str_lnd", 0) or 0)
            s.sig_str_att_against += int(f.get(f"{opp_side}_sig_str_att", 0) or 0)
            s.kd += int(f.get(f"{side}_kd", 0) or 0)
            s.td_lnd += int(f.get(f"{side}_td_lnd", 0) or 0)
            s.td_att += int(f.get(f"{side}_td_att", 0) or 0)
            s.td_lnd_against += int(f.get(f"{opp_side}_td_lnd", 0) or 0)
            s.td_att_against += int(f.get(f"{opp_side}_td_att", 0) or 0)
            s.sub_att += int(f.get(f"{side}_sub_att", 0) or 0)
            s.ctrl_sec += int(f.get(f"{side}_ctrl_sec", 0) or 0)
            s.rev += int(f.get(f"{side}_rev", 0) or 0)

        elo.update(a, b, f["winner"], date)

    feat_df = pd.DataFrame(rows)
    return feat_df, elo, state


# ---------- Synthetic generator (for demo) ----------

def make_synthetic(n_fighters: int = 200, n_fights: int = 4500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"Fighter_{i:03d}" for i in range(n_fighters)]
    # latent skill drives outcome
    skill = rng.normal(1500, 120, size=n_fighters)
    age = rng.integers(22, 38, size=n_fighters)
    height = rng.normal(178, 9, size=n_fighters)
    reach = height + rng.normal(0, 4, size=n_fighters)
    stance = rng.choice(["Orthodox", "Southpaw", "Switch"], size=n_fighters, p=[0.78, 0.18, 0.04])
    weight_classes = list(WEIGHT_CLASS_ORDER.keys())[:9]
    fighter_wc = rng.choice(weight_classes, size=n_fighters)

    start = datetime(2014, 1, 1)
    rows = []
    for i in range(n_fights):
        a_idx, b_idx = rng.choice(n_fighters, 2, replace=False)
        # match similar weight classes more often
        date = (start + timedelta(days=int(i * (365 * 11) / n_fights))).date().isoformat()
        diff = skill[a_idx] - skill[b_idx]
        p_a = 1 / (1 + math.exp(-diff / 150))
        winner = "a" if rng.random() < p_a else "b"
        # tiny skill drift toward winner
        if winner == "a":
            skill[a_idx] += 6; skill[b_idx] -= 6
        else:
            skill[b_idx] += 6; skill[a_idx] -= 6

        # synthetic per-fight stats correlated with skill diff
        base_strikes = 25 + max(0, diff / 8)
        rows.append({
            "event_name": f"UFC Fight Night {i+1}",
            "event_date": date,
            "event_location": "Las Vegas, NV",
            "weight_class": fighter_wc[a_idx],
            "title_bout": 1 if rng.random() < 0.04 else 0,
            "fighter_a": names[a_idx],
            "fighter_b": names[b_idx],
            "winner": winner,
            "method": rng.choice(["Decision - Unanimous", "KO/TKO", "Submission"], p=[0.55, 0.3, 0.15]),
            "round": int(rng.choice([1, 2, 3, 5], p=[0.25, 0.2, 0.5, 0.05])),
            "time": f"{rng.integers(0,5)}:{rng.integers(0,60):02d}",
            "a_kd": int(rng.poisson(0.3 + max(0, diff/300))),
            "b_kd": int(rng.poisson(0.3 + max(0, -diff/300))),
            "a_sig_str_lnd": int(rng.poisson(base_strikes)),
            "a_sig_str_att": int(rng.poisson(base_strikes * 2.2)),
            "b_sig_str_lnd": int(rng.poisson(base_strikes - max(0, diff/12))),
            "b_sig_str_att": int(rng.poisson((base_strikes - max(0, diff/12)) * 2.2)),
            "a_total_str_lnd": 0, "a_total_str_att": 0,
            "b_total_str_lnd": 0, "b_total_str_att": 0,
            "a_td_lnd": int(rng.poisson(0.8)),
            "a_td_att": int(rng.poisson(2.0)),
            "b_td_lnd": int(rng.poisson(0.8)),
            "b_td_att": int(rng.poisson(2.0)),
            "a_sub_att": int(rng.poisson(0.4)),
            "b_sub_att": int(rng.poisson(0.4)),
            "a_rev": int(rng.poisson(0.1)),
            "b_rev": int(rng.poisson(0.1)),
            "a_ctrl_sec": int(rng.integers(0, 360)),
            "b_ctrl_sec": int(rng.integers(0, 360)),
            "a_age": int(age[a_idx]),
            "b_age": int(age[b_idx]),
            "a_height_cm": float(height[a_idx]),
            "b_height_cm": float(height[b_idx]),
            "a_reach_cm": float(reach[a_idx]),
            "b_reach_cm": float(reach[b_idx]),
            "a_stance": stance[a_idx],
            "b_stance": stance[b_idx],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/ufc_fights_raw.csv")
    ap.add_argument("--output", default="../data/features.csv")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate synthetic data instead of reading from --input")
    args = ap.parse_args()

    if args.synthetic:
        log.info("Generating synthetic dataset...")
        raw = make_synthetic()
        Path(args.input).parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.input, index=False)
        log.info("Synthetic raw saved -> %s (%d fights)", args.input, len(raw))
    else:
        raw = pd.read_csv(args.input)

    log.info("Building features for %d fights...", len(raw))
    feat, elo, state = build_features(raw)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(args.output, index=False)

    # Save Elo + state for inference time
    import json
    Path("../models").mkdir(exist_ok=True, parents=True)
    elo_dump = {n: r for n, r in elo.ratings.items()}
    with open("../models/elo_ratings.json", "w") as f:
        json.dump(elo_dump, f, indent=2)

    # Final per-fighter "running stats" snapshot (career totals as of last fight)
    snap = {}
    for name, s in state.items():
        snap[name] = {
            "fights": s.fights, "wins": s.wins, "losses": s.losses,
            "streak": s.streak, "last_5": s.last_5,
            "last_fight_date": s.last_fight_date,
            "sec_in_cage": s.sec_in_cage,
            "sig_str_lnd": s.sig_str_lnd, "sig_str_att": s.sig_str_att,
            "sig_str_lnd_against": s.sig_str_lnd_against,
            "sig_str_att_against": s.sig_str_att_against,
            "kd": s.kd,
            "td_lnd": s.td_lnd, "td_att": s.td_att,
            "td_lnd_against": s.td_lnd_against, "td_att_against": s.td_att_against,
            "sub_att": s.sub_att, "ctrl_sec": s.ctrl_sec, "rev": s.rev,
        }
    with open("../models/fighter_state.json", "w") as f:
        json.dump(snap, f, indent=2)

    log.info("Saved %d feature rows -> %s", len(feat), args.output)
    log.info("Saved Elo ratings + fighter state snapshots to ../models/")


if __name__ == "__main__":
    main()
