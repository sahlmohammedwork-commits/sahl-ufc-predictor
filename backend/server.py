"""
Sahl UFC Predictor - API Server
==========================================
FastAPI app that serves:
  - GET  /api/fighters                    list known fighters with Elo
  - POST /api/predict                     predict a single matchup
  - POST /api/card                        predict a whole card with picks
  - GET  /api/metrics                     model metrics from latest training

Implements:
  - Calibrated XGBoost probabilities
  - American odds <-> implied prob
  - Expected Value (EV)
  - Half-Kelly with bankroll cap (configurable)
"""

from __future__ import annotations
import json
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("api")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "output"
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"


# ---------- Loading model artifacts ----------

class PredictorBundle:
    def __init__(self):
        self.booster = xgb.Booster()
        self.booster.load_model(str(MODEL_DIR / "xgb_model.json"))
        with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)
        with open(MODEL_DIR / "feature_list.json") as f:
            self.feature_list: list[str] = json.load(f)
        with open(MODEL_DIR / "elo_ratings.json") as f:
            self.elo: dict[str, float] = json.load(f)
        with open(MODEL_DIR / "fighter_state.json") as f:
            self.state: dict[str, dict] = json.load(f)
        try:
            with open(OUT_DIR / "metrics.json") as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            self.metrics = {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(X, feature_names=self.feature_list)
        raw = self.booster.predict(d)
        return self.calibrator.transform(raw)


bundle: Optional[PredictorBundle] = None


# ---------- Helpers ----------

WEIGHT_CLASS_ORDER = {
    "Strawweight": 1, "Flyweight": 2, "Bantamweight": 3, "Featherweight": 4,
    "Lightweight": 5, "Welterweight": 6, "Middleweight": 7,
    "Light Heavyweight": 8, "Heavyweight": 9,
    "Women's Strawweight": 1, "Women's Flyweight": 2,
    "Women's Bantamweight": 3, "Women's Featherweight": 4,
}


def safe_div(a, b, default=0.0):
    return a / b if b else default


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def remove_vig_two_way(p_a_implied: float, p_b_implied: float) -> tuple[float, float]:
    s = p_a_implied + p_b_implied
    if s <= 0:
        return 0.5, 0.5
    return p_a_implied / s, p_b_implied / s


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Standard Kelly: f* = (p*b - q) / b, where b = decimal_odds - 1."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, f)


def expected_value(p: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV per unit staked."""
    return p * (decimal_odds - 1.0) * stake - (1.0 - p) * stake


# ---------- Build feature row at inference time ----------

def fighter_running(name: str) -> dict:
    """Return current career-totals dict, with safe defaults."""
    s = bundle.state.get(name)
    if s is None:
        return {
            "fights": 0, "wins": 0, "losses": 0, "streak": 0, "last_5": [],
            "last_fight_date": None, "sec_in_cage": 0,
            "sig_str_lnd": 0, "sig_str_att": 0,
            "sig_str_lnd_against": 0, "sig_str_att_against": 0,
            "kd": 0, "td_lnd": 0, "td_att": 0,
            "td_lnd_against": 0, "td_att_against": 0,
            "sub_att": 0, "ctrl_sec": 0, "rev": 0,
        }
    return s


def build_row(
    a: str, b: str, weight_class: str = "Lightweight",
    title_bout: int = 0,
    a_age: float = 30, b_age: float = 30,
    a_height_cm: float = 178, b_height_cm: float = 178,
    a_reach_cm: float = 183, b_reach_cm: float = 183,
    a_stance: str = "Orthodox", b_stance: str = "Orthodox",
) -> np.ndarray:
    sa, sb = fighter_running(a), fighter_running(b)
    a_min = sa["sec_in_cage"] / 60.0
    b_min = sb["sec_in_cage"] / 60.0

    elo_a = bundle.elo.get(a, 1500.0)
    elo_b = bundle.elo.get(b, 1500.0)

    win_pct_a = safe_div(sa["wins"], sa["fights"], 0.5)
    win_pct_b = safe_div(sb["wins"], sb["fights"], 0.5)
    recent_a = safe_div(sum(sa["last_5"]), len(sa["last_5"]), 0.5) if sa["last_5"] else 0.5
    recent_b = safe_div(sum(sb["last_5"]), len(sb["last_5"]), 0.5) if sb["last_5"] else 0.5

    slpm_a = safe_div(sa["sig_str_lnd"], a_min); slpm_b = safe_div(sb["sig_str_lnd"], b_min)
    sapm_a = safe_div(sa["sig_str_lnd_against"], a_min); sapm_b = safe_div(sb["sig_str_lnd_against"], b_min)
    str_acc_a = safe_div(sa["sig_str_lnd"], sa["sig_str_att"], 0.4)
    str_acc_b = safe_div(sb["sig_str_lnd"], sb["sig_str_att"], 0.4)
    str_def_a = 1.0 - safe_div(sa["sig_str_lnd_against"], sa["sig_str_att_against"], 0.6)
    str_def_b = 1.0 - safe_div(sb["sig_str_lnd_against"], sb["sig_str_att_against"], 0.6)
    kd15_a = safe_div(sa["kd"], a_min) * 15.0
    kd15_b = safe_div(sb["kd"], b_min) * 15.0
    td15_a = safe_div(sa["td_lnd"], a_min) * 15.0
    td15_b = safe_div(sb["td_lnd"], b_min) * 15.0
    td_acc_a = safe_div(sa["td_lnd"], sa["td_att"], 0.3)
    td_acc_b = safe_div(sb["td_lnd"], sb["td_att"], 0.3)
    td_def_a = 1.0 - safe_div(sa["td_lnd_against"], sa["td_att_against"], 0.7)
    td_def_b = 1.0 - safe_div(sb["td_lnd_against"], sb["td_att_against"], 0.7)
    ctrl_pct_a = safe_div(sa["ctrl_sec"], sa["sec_in_cage"])
    ctrl_pct_b = safe_div(sb["ctrl_sec"], sb["sec_in_cage"])
    sub15_a = safe_div(sa["sub_att"], a_min) * 15.0
    sub15_b = safe_div(sb["sub_att"], b_min) * 15.0
    getup_a = safe_div(sa["rev"], a_min)
    getup_b = safe_div(sb["rev"], b_min)

    stance_match = int(("orthodox" in a_stance.lower()) ^ ("orthodox" in b_stance.lower()))
    layoff_a = 180; layoff_b = 180  # default if unknown
    wc_enc = WEIGHT_CLASS_ORDER.get(weight_class, 5)

    feats = {
        "elo_diff": elo_a - elo_b,
        "elo_trend_diff": 0.0,
        "win_streak_diff": sa["streak"] - sb["streak"],
        "win_pct_diff": win_pct_a - win_pct_b,
        "recent_form_diff": recent_a - recent_b,
        "slpm_diff": slpm_a - slpm_b,
        "sapm_diff": sapm_b - sapm_a,
        "str_acc_diff": str_acc_a - str_acc_b,
        "str_def_diff": str_def_a - str_def_b,
        "kd_per15_diff": kd15_a - kd15_b,
        "td_per15_diff": td15_a - td15_b,
        "td_acc_diff": td_acc_a - td_acc_b,
        "td_def_diff": td_def_a - td_def_b,
        "ctrl_pct_diff": ctrl_pct_a - ctrl_pct_b,
        "sub_att_per15_diff": sub15_a - sub15_b,
        "getup_speed_diff": getup_a - getup_b,
        "age_diff": a_age - b_age,
        "height_diff_cm": a_height_cm - b_height_cm,
        "reach_diff_cm": a_reach_cm - b_reach_cm,
        "weight_class_encoded": wc_enc,
        "stance_match": stance_match,
        "layoff_diff_days": layoff_a - layoff_b,
        "ufc_exp_diff": sa["fights"] - sb["fights"],
        "title_bout": title_bout,
    }
    return np.array([[feats[k] for k in bundle.feature_list]], dtype=np.float32)


# ---------- Pydantic schemas ----------

class FightInput(BaseModel):
    fighter_a: str
    fighter_b: str
    weight_class: str = "Lightweight"
    title_bout: int = 0
    a_age: float = 30; b_age: float = 30
    a_height_cm: float = 178; b_height_cm: float = 178
    a_reach_cm: float = 183; b_reach_cm: float = 183
    a_stance: str = "Orthodox"; b_stance: str = "Orthodox"
    odds_a: Optional[int] = None  # American odds, e.g. -150
    odds_b: Optional[int] = None
    bankroll: float = 1000.0
    kelly_fraction_cap: float = 0.5  # 0.5 = half-Kelly
    bankroll_cap_pct: float = 0.05   # max 5% of bankroll on a single bet


class CardInput(BaseModel):
    fights: list[FightInput]
    bankroll: float = 1000.0


class PredictionOut(BaseModel):
    fighter_a: str
    fighter_b: str
    p_a: float
    p_b: float
    elo_a: float
    elo_b: float
    pick: str
    confidence: float
    market_p_a: Optional[float] = None
    market_p_b: Optional[float] = None
    edge_a: Optional[float] = None
    edge_b: Optional[float] = None
    ev_a: Optional[float] = None
    ev_b: Optional[float] = None
    kelly_stake_a: Optional[float] = None
    kelly_stake_b: Optional[float] = None
    recommendation: str = ""
    explanations: list[dict] = Field(default_factory=list)


# ---------- App ----------

app = FastAPI(title="Sahl UFC Predictor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _load():
    global bundle
    bundle = PredictorBundle()
    log.info("Loaded model. %d fighters in state, %d in Elo.",
             len(bundle.state), len(bundle.elo))


@app.get("/api/health")
def health():
    return {"status": "ok", "fighters_known": len(bundle.state)}


@app.get("/api/fighters")
def fighters(limit: int = 5000):
    """Returns known fighters sorted by Elo desc."""
    items = sorted(
        ((n, r) for n, r in bundle.elo.items()),
        key=lambda x: -x[1],
    )[:limit]
    out = []
    for name, r in items:
        s = bundle.state.get(name, {})
        out.append({
            "name": name, "elo": round(r, 1),
            "fights": s.get("fights", 0),
            "wins": s.get("wins", 0), "losses": s.get("losses", 0),
            "streak": s.get("streak", 0),
        })
    return out


@app.get("/api/metrics")
def metrics():
    return bundle.metrics or {"note": "Train the model first."}


@app.get("/api/upcoming")
def upcoming():
    """Fetch the next scheduled UFC event + matchups from ufcstats.com."""
    try:
        from upcoming import get_upcoming_card
        return get_upcoming_card()
    except Exception as e:
        log.warning("Upcoming fetch failed: %s", e)
        return {"event": None, "fights": [], "error": str(e)}


def _predict_one(f: FightInput) -> PredictionOut:
    X = build_row(
        f.fighter_a, f.fighter_b, f.weight_class, f.title_bout,
        f.a_age, f.b_age, f.a_height_cm, f.b_height_cm,
        f.a_reach_cm, f.b_reach_cm, f.a_stance, f.b_stance,
    )
    p_a = float(bundle.predict_proba(X)[0])
    p_a = float(np.clip(p_a, 0.01, 0.99))
    p_b = 1 - p_a

    elo_a = bundle.elo.get(f.fighter_a, 1500.0)
    elo_b = bundle.elo.get(f.fighter_b, 1500.0)

    out = PredictionOut(
        fighter_a=f.fighter_a, fighter_b=f.fighter_b,
        p_a=p_a, p_b=p_b, elo_a=elo_a, elo_b=elo_b,
        pick=f.fighter_a if p_a >= p_b else f.fighter_b,
        confidence=max(p_a, p_b),
    )

    # Market & EV
    if f.odds_a is not None and f.odds_b is not None:
        imp_a = american_to_implied(f.odds_a)
        imp_b = american_to_implied(f.odds_b)
        m_a, m_b = remove_vig_two_way(imp_a, imp_b)
        dec_a, dec_b = american_to_decimal(f.odds_a), american_to_decimal(f.odds_b)

        edge_a = p_a - m_a
        edge_b = p_b - m_b
        ev_a = expected_value(p_a, dec_a)
        ev_b = expected_value(p_b, dec_b)
        f_a = kelly_fraction(p_a, dec_a) * f.kelly_fraction_cap
        f_b = kelly_fraction(p_b, dec_b) * f.kelly_fraction_cap
        f_a = min(f_a, f.bankroll_cap_pct)
        f_b = min(f_b, f.bankroll_cap_pct)
        stake_a = round(f_a * f.bankroll, 2)
        stake_b = round(f_b * f.bankroll, 2)

        out.market_p_a, out.market_p_b = m_a, m_b
        out.edge_a, out.edge_b = edge_a, edge_b
        out.ev_a, out.ev_b = ev_a, ev_b
        out.kelly_stake_a, out.kelly_stake_b = stake_a, stake_b

        # Recommendation
        if ev_a > 0.02 and stake_a > 0:
            out.recommendation = (
                f"BET {f.fighter_a} - stake ${stake_a:.2f} "
                f"(EV {ev_a*100:+.1f}%, edge {edge_a*100:+.1f} pts)"
            )
        elif ev_b > 0.02 and stake_b > 0:
            out.recommendation = (
                f"BET {f.fighter_b} - stake ${stake_b:.2f} "
                f"(EV {ev_b*100:+.1f}%, edge {edge_b*100:+.1f} pts)"
            )
        else:
            out.recommendation = "PASS - no positive-EV side"

    # Real per-prediction SHAP-style contributions via XGBoost's pred_contribs.
    try:
        d = xgb.DMatrix(X, feature_names=bundle.feature_list)
        contribs_arr = bundle.booster.predict(d, pred_contribs=True)[0]
        # Last column is the bias; rest map to feature_list in order.
        contribs = []
        for i, name in enumerate(bundle.feature_list):
            contribs.append({
                "feature": name,
                "value": float(X[0, i]),
                "shap": float(contribs_arr[i]),
            })
        contribs.sort(key=lambda x: -abs(x["shap"]))
        out.explanations = contribs[:6]
    except Exception as e:
        log.warning("Explanation step failed: %s", e)

    return out


@app.post("/api/predict", response_model=PredictionOut)
def predict(f: FightInput):
    if bundle is None:
        raise HTTPException(503, "Model not loaded")
    return _predict_one(f)


@app.post("/api/card")
def predict_card(card: CardInput):
    """Predicts whole card and returns picks ranked by EV."""
    if bundle is None:
        raise HTTPException(503, "Model not loaded")
    preds = []
    for f in card.fights:
        f.bankroll = card.bankroll
        preds.append(_predict_one(f))
    # Best EV side per fight
    ranked = []
    for p in preds:
        best_ev = None
        if p.ev_a is not None:
            best_ev = max(p.ev_a, p.ev_b)
        ranked.append({"prediction": p.dict(), "best_ev": best_ev or 0.0})
    ranked.sort(key=lambda x: -x["best_ev"])
    return {"card": ranked, "n": len(ranked)}


# ---------- Static frontend ----------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
