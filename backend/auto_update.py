"""
Sahl UFC Predictor - Auto-Update Worker
========================================
Scrapes any UFC events that happened since our last training, merges them
into the dataset, retrains the model, and (because Render auto-deploys
when files change) the live site picks up the new model.

Designed to be run as a Render Cron Job, weekly (every Monday 8am UTC).

Flow:
  1. Read last-event-date from data/last_update.json (or use 2 years ago)
  2. Scrape only NEW events from ufcstats.com (Render's network is unblocked)
  3. Append to data/ufc_fights_raw.csv
  4. Re-run features.py + train.py
  5. Commit + push to GitHub via the GIT_TOKEN env var
  6. Render auto-redeploys with the new model

Env vars needed on Render:
  GIT_REPO       - e.g. https://github.com/sahlmohammedwork-commits/sahl-ufc-predictor.git
  GIT_TOKEN      - GitHub personal access token with `repo` scope
  GIT_USER_EMAIL - your email
  GIT_USER_NAME  - your name or username
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("auto-update")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "ufc_fights_raw.csv"
STATE = DATA / "last_update.json"


def get_last_date() -> str:
    if STATE.exists():
        with open(STATE) as f:
            return json.load(f).get("last_event_date",
                                    (datetime.utcnow() - timedelta(days=730)).date().isoformat())
    if RAW.exists():
        df = pd.read_csv(RAW, usecols=["event_date"])
        if not df.empty:
            return df["event_date"].max()
    return (datetime.utcnow() - timedelta(days=730)).date().isoformat()


def scrape_since(cutoff_date: str) -> pd.DataFrame:
    """Use scraper.py functions directly to fetch only new fights."""
    sys.path.insert(0, str(ROOT / "backend"))
    from scraper import Fetcher, list_events, list_fights_for_event, parse_fight, FightRow
    from dataclasses import asdict

    fetcher = Fetcher(delay=1.0)
    events = list_events(fetcher, since_date=cutoff_date)
    log.info("Found %d new events since %s", len(events), cutoff_date)
    rows = []
    for ev in events:
        try:
            for fu in list_fights_for_event(fetcher, ev["url"]):
                try:
                    row = parse_fight(fetcher, fu, ev)
                    if row:
                        rows.append(asdict(row))
                except Exception as e:
                    log.warning("Fight skipped: %s", e)
        except Exception as e:
            log.warning("Event skipped: %s", e)
    return pd.DataFrame(rows)


def git_push():
    repo = os.environ.get("GIT_REPO")
    token = os.environ.get("GIT_TOKEN")
    email = os.environ.get("GIT_USER_EMAIL", "auto@sahl-ufc.local")
    name = os.environ.get("GIT_USER_NAME", "sahl-auto")
    if not (repo and token):
        log.warning("GIT_REPO/GIT_TOKEN not set - skipping push")
        return
    repo_with_token = repo.replace("https://", f"https://{token}@")

    cmds = [
        ["git", "config", "user.email", email],
        ["git", "config", "user.name", name],
        ["git", "remote", "set-url", "origin", repo_with_token],
        ["git", "add", "data/", "models/", "output/"],
        ["git", "commit", "-m", f"auto-update {datetime.utcnow().date().isoformat()}"],
        ["git", "push", "origin", "main"],
    ]
    for cmd in cmds:
        log.info("Running: %s", " ".join(c if "@" not in c else "<token>" for c in cmd))
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0 and "nothing to commit" not in result.stdout + result.stderr:
            log.warning("Cmd output: %s\n%s", result.stdout, result.stderr)


def main():
    cutoff = get_last_date()
    log.info("Last training cutoff: %s", cutoff)

    new_df = scrape_since(cutoff)
    if new_df.empty:
        log.info("No new fights. Exiting.")
        return

    if RAW.exists():
        old_df = pd.read_csv(RAW)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["fight_url"], keep="last")
    else:
        merged = new_df
    merged.to_csv(RAW, index=False)
    log.info("Total fights now: %d (added %d)", len(merged), len(new_df))

    new_cutoff = merged["event_date"].max()
    STATE.write_text(json.dumps({"last_event_date": new_cutoff,
                                 "updated_at": datetime.utcnow().isoformat()}))

    log.info("Rebuilding features...")
    subprocess.run([sys.executable, "features.py"], cwd=str(ROOT / "backend"), check=True)
    log.info("Retraining model...")
    subprocess.run([sys.executable, "train.py"], cwd=str(ROOT / "backend"), check=True)

    git_push()
    log.info("Auto-update complete. Render will redeploy momentarily.")


if __name__ == "__main__":
    main()
