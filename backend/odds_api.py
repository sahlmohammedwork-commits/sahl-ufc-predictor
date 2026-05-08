"""
Sahl UFC Predictor - Live Odds Module
======================================
Fetches upcoming UFC fights AND their current sportsbook odds from
The Odds API (https://the-odds-api.com).

Free tier: 500 requests/month. We cache responses for 1 hour to stay
well under the limit. One scheduled card refresh = 1 request.

Set the API key as the ODDS_API_KEY environment variable on Render.
Get a free key at https://the-odds-api.com/#get-access
"""

from __future__ import annotations
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("odds_api")

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "odds_cache.json"
CACHE_TTL_SEC = 3600  # 1 hour


def _load_cache() -> Optional[dict]:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        if time.time() - data.get("fetched_at", 0) < CACHE_TTL_SEC:
            return data
    except Exception:
        pass
    return None


def _save_cache(data: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception as e:
        log.warning("Failed to write odds cache: %s", e)


def fetch_upcoming_with_odds() -> dict:
    """
    Returns dict with keys:
      - event: {name, date, location} for the next event (best-effort)
      - fights: list of {fighter_a, fighter_b, weight_class, odds_a, odds_b, commence_time}
      - error: str if something failed
      - source: "live" | "cache" | "none"
    """
    cached = _load_cache()
    if cached:
        log.info("Using cached odds (age: %ds)", int(time.time() - cached.get("fetched_at", 0)))
        return {**cached["payload"], "source": "cache"}

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        return {
            "event": None,
            "fights": [],
            "error": "ODDS_API_KEY env var not set on Render. Get a free key at the-odds-api.com",
            "source": "none",
        }

    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "apiKey": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 401:
            return {"event": None, "fights": [], "error": "Invalid ODDS_API_KEY", "source": "none"}
        if r.status_code == 429:
            return {"event": None, "fights": [], "error": "Rate limit hit (500/mo on free tier)", "source": "none"}
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        log.warning("Odds API fetch failed: %s", e)
        return {"event": None, "fights": [], "error": f"Fetch failed: {e}", "source": "none"}

    if not events:
        return {"event": None, "fights": [], "error": "No upcoming UFC events", "source": "live"}

    # Sort by commence_time ascending — earliest first
    events.sort(key=lambda e: e.get("commence_time", ""))

    fights = []
    earliest_date = events[0].get("commence_time", "")[:10] if events else ""
    same_day_events = [e for e in events if e.get("commence_time", "")[:10] == earliest_date]

    for ev in same_day_events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        commence = ev.get("commence_time", "")

        # Average odds across all bookmakers for this fight
        prices_home = []
        prices_away = []
        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if name == home:
                        prices_home.append(price)
                    elif name == away:
                        prices_away.append(price)

        odds_a = round(sum(prices_home) / len(prices_home)) if prices_home else None
        odds_b = round(sum(prices_away) / len(prices_away)) if prices_away else None

        fights.append({
            "fighter_a": home,
            "fighter_b": away,
            "weight_class": "Lightweight",  # Odds API doesn't return weight class
            "odds_a": odds_a,
            "odds_b": odds_b,
            "commence_time": commence,
            "num_books": max(len(prices_home), len(prices_away)),
        })

    payload = {
        "event": {
            "name": f"UFC Card · {earliest_date}",
            "date": earliest_date,
            "location": "Multiple sportsbooks",
        },
        "fights": fights,
        "error": None,
    }

    _save_cache({"fetched_at": time.time(), "payload": payload})
    return {**payload, "source": "live"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_upcoming_with_odds()
    print(json.dumps(result, indent=2))
