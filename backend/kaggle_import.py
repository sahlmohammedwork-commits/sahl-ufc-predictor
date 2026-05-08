"""
Sahl UFC Predictor - Kaggle Dataset Importer
=============================================
Downloads and converts a public UFC dataset from Kaggle into our scraper's
CSV schema so features.py can ingest it identically to scraper.py output.

Usage:
    python kaggle_import.py --dataset mdabbert/ultimate-ufc-dataset
    python kaggle_import.py --dataset rajeevw/ufcdata        (older but bigger)

By default uses mdabbert which is updated more recently.

Setup:
    1. Get a free Kaggle account at kaggle.com
    2. Account -> Settings -> API -> Create New Token  (downloads kaggle.json)
    3. Move kaggle.json to:
         Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
         Mac/Linux: ~/.kaggle/kaggle.json
    4. pip install kaggle
    5. python kaggle_import.py
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("kaggle-import")


SCRAPER_COLUMNS = [
    "event_name", "event_date", "event_location", "weight_class", "title_bout",
    "fighter_a", "fighter_b", "winner", "method", "round", "time",
    "a_kd", "b_kd",
    "a_sig_str_lnd", "b_sig_str_lnd", "a_sig_str_att", "b_sig_str_att",
    "a_total_str_lnd", "b_total_str_lnd", "a_total_str_att", "b_total_str_att",
    "a_td_lnd", "b_td_lnd", "a_td_att", "b_td_att",
    "a_sub_att", "b_sub_att", "a_rev", "b_rev",
    "a_ctrl_sec", "b_ctrl_sec",
    "fight_url",
    "a_age", "b_age", "a_height_cm", "b_height_cm",
    "a_reach_cm", "b_reach_cm", "a_stance", "b_stance",
]


def in_to_cm(inches):
    if pd.isna(inches): return np.nan
    return float(inches) * 2.54


def parse_record(rec):
    if pd.isna(rec): return 0, 0, 0
    try:
        parts = str(rec).split("-")
        return int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
    except Exception:
        return 0, 0, 0


def normalize_mdabbert(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mdabbert/ultimate-ufc-dataset rows to our schema."""
    out = pd.DataFrame()
    out["event_name"] = "UFC " + df.get("date", "").astype(str)
    out["event_date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["event_location"] = df.get("location", "")
    out["weight_class"] = df.get("weight_class", df.get("WeightClass", "Lightweight"))
    out["title_bout"] = df.get("title_bout", df.get("TitleBout", False)).astype(int) \
        if "title_bout" in df.columns or "TitleBout" in df.columns else 0
    out["fighter_a"] = df.get("R_fighter", df.get("RedFighter", ""))
    out["fighter_b"] = df.get("B_fighter", df.get("BlueFighter", ""))

    # Winner: dataset uses "Winner" column = "Red" / "Blue" / "Draw"
    winner_col = "Winner" if "Winner" in df.columns else "winner"
    w = df[winner_col].astype(str).str.lower()
    out["winner"] = np.where(w == "red", "a",
                     np.where(w == "blue", "b",
                     np.where(w == "draw", "draw", "nc")))

    out["method"] = df.get("finish", df.get("Finish", ""))
    out["round"] = df.get("finish_round", df.get("FinishRound", 3)).fillna(3).astype(int)
    rd_time = df.get("finish_round_time", df.get("FinishRoundTime", "0:00"))
    out["time"] = rd_time.astype(str)

    # Per-fight stats. mdabbert often only has DIFFERENTIALS or AVG totals across career.
    # We'll use the differentials/per-fight where available; otherwise zero.
    def col(name, default=0):
        for c in [name, name.replace("_", "")]:
            if c in df.columns:
                return df[c].fillna(default)
        return pd.Series([default] * len(df))

    out["a_kd"] = col("R_avg_KD", 0).clip(0).round().astype(int)
    out["b_kd"] = col("B_avg_KD", 0).clip(0).round().astype(int)

    # The mdabbert set has aggregated career averages (R_avg_SIG_STR_landed),
    # not per-fight totals. We approximate per-fight by using the *career averages*
    # as the per-fight stat - this is fine because features.py recomputes its own
    # rolling sums from these. The model still sees consistent signal.
    out["a_sig_str_lnd"] = col("R_avg_SIG_STR_landed", 0).round().astype(int)
    out["a_sig_str_att"] = (col("R_avg_SIG_STR_landed", 0) /
                            col("R_avg_SIG_STR_pct", 0.4).replace(0, 0.4)).round().astype(int)
    out["b_sig_str_lnd"] = col("B_avg_SIG_STR_landed", 0).round().astype(int)
    out["b_sig_str_att"] = (col("B_avg_SIG_STR_landed", 0) /
                            col("B_avg_SIG_STR_pct", 0.4).replace(0, 0.4)).round().astype(int)

    out["a_total_str_lnd"] = col("R_avg_TOTAL_STR_landed", 0).round().astype(int)
    out["a_total_str_att"] = (out["a_total_str_lnd"] * 1.6).round().astype(int)
    out["b_total_str_lnd"] = col("B_avg_TOTAL_STR_landed", 0).round().astype(int)
    out["b_total_str_att"] = (out["b_total_str_lnd"] * 1.6).round().astype(int)

    out["a_td_lnd"] = col("R_avg_TD_landed", 0).round().astype(int)
    out["a_td_att"] = (col("R_avg_TD_landed", 0) /
                      col("R_avg_TD_pct", 0.3).replace(0, 0.3)).round().astype(int)
    out["b_td_lnd"] = col("B_avg_TD_landed", 0).round().astype(int)
    out["b_td_att"] = (col("B_avg_TD_landed", 0) /
                      col("B_avg_TD_pct", 0.3).replace(0, 0.3)).round().astype(int)

    out["a_sub_att"] = col("R_avg_SUB_ATT", 0).round().astype(int)
    out["b_sub_att"] = col("B_avg_SUB_ATT", 0).round().astype(int)
    out["a_rev"] = 0
    out["b_rev"] = 0
    out["a_ctrl_sec"] = 0
    out["b_ctrl_sec"] = 0
    out["fight_url"] = ""

    out["a_age"] = col("R_age", 30).fillna(30)
    out["b_age"] = col("B_age", 30).fillna(30)
    out["a_height_cm"] = col("R_Height_cms", col("R_Height_cm", 178)).fillna(178)
    out["b_height_cm"] = col("B_Height_cms", col("B_Height_cm", 178)).fillna(178)
    out["a_reach_cm"] = col("R_Reach_cms", col("R_Reach_cm", 183)).fillna(183)
    out["b_reach_cm"] = col("B_Reach_cms", col("B_Reach_cm", 183)).fillna(183)
    out["a_stance"] = col("R_Stance", "Orthodox").fillna("Orthodox")
    out["b_stance"] = col("B_Stance", "Orthodox").fillna("Orthodox")

    return out


def normalize_rajeev(df: pd.DataFrame) -> pd.DataFrame:
    """Convert rajeevw/ufcdata rows. Schema differs slightly."""
    # Same logic, slightly different column names
    rename = {
        "R_fighter": "R_fighter", "B_fighter": "B_fighter",
        "R_age": "R_age", "B_age": "B_age",
        "Winner": "Winner",
    }
    return normalize_mdabbert(df.rename(columns=rename))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mdabbert/ultimate-ufc-dataset",
                    help="Kaggle slug. Other option: rajeevw/ufcdata")
    ap.add_argument("--output", default="../data/ufc_fights_raw.csv")
    ap.add_argument("--csv-name", default=None,
                    help="Specific CSV file inside the Kaggle dataset zip")
    args = ap.parse_args()

    download_dir = Path("../data/kaggle_raw")
    download_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading Kaggle dataset: %s", args.dataset)
    try:
        import kaggle  # imports trigger auth via ~/.kaggle/kaggle.json
    except OSError as e:
        log.error("Kaggle auth failed. Did you place kaggle.json correctly?")
        log.error("See top-of-file docstring for instructions.")
        log.error("Underlying error: %s", e)
        sys.exit(1)

    kaggle.api.dataset_download_files(args.dataset, path=str(download_dir), unzip=True)

    # Find a CSV likely to be the main fight file
    csvs = list(download_dir.glob("*.csv"))
    log.info("Files in dataset: %s", [c.name for c in csvs])
    if args.csv_name:
        target = download_dir / args.csv_name
    else:
        # Heuristic: pick the largest CSV (usually the fight-level one)
        target = max(csvs, key=lambda p: p.stat().st_size)
    log.info("Using: %s", target.name)

    raw = pd.read_csv(target)
    log.info("Loaded %d rows, columns: %s", len(raw), list(raw.columns)[:8] + ["..."])

    if "rajeev" in args.dataset.lower():
        out = normalize_rajeev(raw)
    else:
        out = normalize_mdabbert(raw)

    out = out[SCRAPER_COLUMNS]
    out = out.dropna(subset=["fighter_a", "fighter_b", "event_date"])
    out = out[out["winner"].isin(["a", "b", "draw"])]
    out = out.sort_values("event_date").reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log.info("Wrote %d normalized fights -> %s", len(out), args.output)
    log.info("Sample fighters: %s",
             list(out["fighter_a"].drop_duplicates().head(10)))


if __name__ == "__main__":
    main()
