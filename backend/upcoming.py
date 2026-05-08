"""
Sahl UFC Predictor - Upcoming Card Scraper
==========================================
Fetches the next scheduled UFC event + matchups from ufcstats.com
so the website can show real upcoming fights instantly.

Used by the API: GET /api/upcoming
"""

from __future__ import annotations
import logging
import re
import sys
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE = "https://www.ufcstats.com"
UPCOMING_URL = f"{BASE}/statistics/events/upcoming"
HEADERS = {"User-Agent": "SahlUFCPredictor/1.0 (Research)"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("upcoming")


def fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    backoff = 1.5
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            log.warning("Fetch %s failed (try %d): %s", url, i + 1, e)
        time.sleep(backoff)
        backoff *= 2
    return None


def get_next_event() -> Optional[dict]:
    """Returns {name, date, location, url} for the next upcoming UFC event."""
    soup = fetch(UPCOMING_URL)
    if not soup:
        return None
    rows = soup.select("tr.b-statistics__table-row")
    for row in rows:
        a = row.select_one("a.b-link")
        if not a:
            continue
        name = a.get_text(strip=True)
        url = a.get("href", "").strip()
        date_el = row.select_one("span.b-statistics__date")
        date_str = date_el.get_text(strip=True) if date_el else ""
        cols = row.select("td.b-statistics__table-col")
        location = cols[1].get_text(strip=True) if len(cols) > 1 else ""
        return {"name": name, "date": date_str, "location": location, "url": url}
    return None


def get_event_fights(event_url: str) -> list[dict]:
    """Returns list of {fighter_a, fighter_b, weight_class}."""
    soup = fetch(event_url)
    if not soup:
        return []
    fights = []
    for tr in soup.select("tr.b-fight-details__table-row"):
        names = tr.select("a.b-link")
        if len(names) < 2:
            continue
        a_name = names[0].get_text(strip=True)
        b_name = names[1].get_text(strip=True)
        cols = tr.select("td.b-fight-details__table-col")
        # find weight-class column
        wc = ""
        for c in cols:
            txt = c.get_text(" ", strip=True)
            if any(k in txt.lower() for k in [
                "weight", "lightweight", "welterweight", "middleweight",
                "heavyweight", "featherweight", "bantamweight", "flyweight",
                "strawweight",
            ]):
                wc = re.sub(r"\s+", " ", txt).strip()
                break
        fights.append({
            "fighter_a": a_name,
            "fighter_b": b_name,
            "weight_class": wc or "Lightweight",
        })
    return fights


def get_upcoming_card() -> dict:
    """Convenience: next event + its fights."""
    event = get_next_event()
    if not event:
        return {"event": None, "fights": []}
    fights = get_event_fights(event["url"])
    return {"event": event, "fights": fights}


if __name__ == "__main__":
    import json
    print(json.dumps(get_upcoming_card(), indent=2))
