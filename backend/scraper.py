"""
Sahl UFC Predictor - ufcstats.com Scraper
==========================================
Scrapes complete fight history from ufcstats.com.

USAGE:
    python scraper.py --output ../data/ufc_fights_raw.csv --years 10

What it grabs per fight:
    - Event date, location, title bout flag, weight class
    - Both fighters, winner, method, round, time
    - Per-fight stats: sig strikes (att/lnd, by target & position),
      total strikes, takedowns (att/lnd), submission attempts,
      reversals, control time, knockdowns

Be polite: 1 req/sec, retries with backoff.
"""

from __future__ import annotations
import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.ufcstats.com"
EVENTS_URL = f"{BASE}/statistics/events/completed?page=all"
HEADERS = {
    "User-Agent": (
        "SahlUFCPredictor/1.0 (Research; respectful scraper, 1 req/sec)"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scraper")


# ---------- HTTP ----------

class Fetcher:
    def __init__(self, delay: float = 1.0, max_retries: int = 4):
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last = 0.0

    def get(self, url: str) -> BeautifulSoup:
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        backoff = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, timeout=20)
                self._last = time.time()
                if r.status_code == 200:
                    return BeautifulSoup(r.text, "html.parser")
                log.warning("HTTP %s on %s (attempt %s)", r.status_code, url, attempt)
            except requests.RequestException as e:
                log.warning("Error fetching %s: %s (attempt %s)", url, e, attempt)
            time.sleep(backoff)
            backoff *= 2
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts")


# ---------- Data classes ----------

@dataclass
class FightRow:
    event_name: str = ""
    event_date: str = ""
    event_location: str = ""
    weight_class: str = ""
    title_bout: int = 0
    fighter_a: str = ""
    fighter_b: str = ""
    winner: str = ""              # "a" / "b" / "draw" / "nc"
    method: str = ""
    round: int = 0
    time: str = ""
    # per-fight totals (a / b)
    a_kd: int = 0; b_kd: int = 0
    a_sig_str_lnd: int = 0; b_sig_str_lnd: int = 0
    a_sig_str_att: int = 0; b_sig_str_att: int = 0
    a_total_str_lnd: int = 0; b_total_str_lnd: int = 0
    a_total_str_att: int = 0; b_total_str_att: int = 0
    a_td_lnd: int = 0; b_td_lnd: int = 0
    a_td_att: int = 0; b_td_att: int = 0
    a_sub_att: int = 0; b_sub_att: int = 0
    a_rev: int = 0; b_rev: int = 0
    a_ctrl_sec: int = 0; b_ctrl_sec: int = 0
    fight_url: str = ""


# ---------- Parsing helpers ----------

def parse_x_of_y(s: str) -> tuple[int, int]:
    """'42 of 87' -> (42, 87). '---' -> (0, 0)."""
    s = (s or "").strip()
    m = re.match(r"(\d+)\s*of\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if s.isdigit():
        return int(s), int(s)
    return 0, 0


def parse_int(s: str) -> int:
    s = (s or "").strip().replace("---", "0").replace("--", "0")
    return int(s) if s.isdigit() else 0


def parse_ctrl_time(s: str) -> int:
    s = (s or "").strip()
    if not s or s in ("---", "--"):
        return 0
    m = re.match(r"(\d+):(\d+)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def parse_event_date(s: str) -> str:
    """ 'November 11, 2023' -> '2023-11-11' """
    s = s.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


# ---------- Scrapers ----------

def list_events(fetcher: Fetcher, since_date: Optional[str] = None) -> list[dict]:
    """Returns [{'name','date','location','url'}, ...] - newest first."""
    soup = fetcher.get(EVENTS_URL)
    rows = soup.select("tr.b-statistics__table-row")
    out: list[dict] = []
    for row in rows:
        a = row.select_one("a.b-link")
        if not a:
            continue
        name = a.get_text(strip=True)
        url = a.get("href", "").strip()
        date_el = row.select_one("span.b-statistics__date")
        date_str = parse_event_date(date_el.get_text(strip=True)) if date_el else ""
        loc_el = row.select("td.b-statistics__table-col")
        location = loc_el[1].get_text(strip=True) if len(loc_el) > 1 else ""
        if since_date and date_str and date_str < since_date:
            continue
        out.append({"name": name, "date": date_str, "location": location, "url": url})
    return out


def list_fights_for_event(fetcher: Fetcher, event_url: str) -> list[str]:
    soup = fetcher.get(event_url)
    fight_urls = []
    for tr in soup.select("tr.b-fight-details__table-row"):
        onclick = tr.get("data-link") or ""
        if onclick:
            fight_urls.append(onclick)
    return fight_urls


def parse_fight(fetcher: Fetcher, fight_url: str, event: dict) -> Optional[FightRow]:
    soup = fetcher.get(fight_url)

    # Fighters
    persons = soup.select("div.b-fight-details__person")
    if len(persons) < 2:
        return None
    fighter_a = persons[0].select_one("h3.b-fight-details__person-name").get_text(strip=True)
    fighter_b = persons[1].select_one("h3.b-fight-details__person-name").get_text(strip=True)

    # Result
    status_a = persons[0].select_one("i.b-fight-details__person-status").get_text(strip=True).upper()
    status_b = persons[1].select_one("i.b-fight-details__person-status").get_text(strip=True).upper()
    if status_a == "W":
        winner = "a"
    elif status_b == "W":
        winner = "b"
    elif status_a == "D" or status_b == "D":
        winner = "draw"
    else:
        winner = "nc"

    # Header bits
    head = soup.select_one("i.b-fight-details__fight-title")
    title_text = head.get_text(strip=True) if head else ""
    title_bout = 1 if "title" in title_text.lower() else 0
    weight_class = re.sub(r"(?i)\s*ufc\s*", "", title_text)
    weight_class = re.sub(r"(?i)title\s*bout|interim", "", weight_class).strip()

    # Method / round / time
    method = round_no = time_str = ""
    info_rows = soup.select("p.b-fight-details__text")
    for p in info_rows:
        text = p.get_text(" ", strip=True)
        m_meth = re.search(r"Method:\s*([^|]+?)(?:Round:|$)", text)
        m_round = re.search(r"Round:\s*(\d+)", text)
        m_time = re.search(r"Time:\s*(\d+:\d+)", text)
        if m_meth and not method:
            method = m_meth.group(1).strip()
        if m_round and not round_no:
            round_no = m_round.group(1)
        if m_time and not time_str:
            time_str = m_time.group(1)

    row = FightRow(
        event_name=event["name"],
        event_date=event["date"],
        event_location=event["location"],
        weight_class=weight_class,
        title_bout=title_bout,
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        winner=winner,
        method=method,
        round=int(round_no) if str(round_no).isdigit() else 0,
        time=time_str,
        fight_url=fight_url,
    )

    # Totals table - first 'b-fight-details__table' is totals
    tables = soup.select("table.b-fight-details__table")
    if not tables:
        return row
    totals = tables[0]
    cells = totals.select("td.b-fight-details__table-col")
    # Each cell has two <p> stacked: top = fighter A's value, bottom = B
    def cell_pair(idx: int) -> tuple[str, str]:
        if idx >= len(cells):
            return "", ""
        ps = cells[idx].select("p")
        return (ps[0].get_text(strip=True) if ps else "",
                ps[1].get_text(strip=True) if len(ps) > 1 else "")

    # Column order on UFC stats totals:
    # 0 Fighter | 1 KD | 2 Sig.str. | 3 Sig.str.% | 4 Total str. | 5 TD | 6 TD% | 7 Sub.att | 8 Rev. | 9 Ctrl
    a_kd, b_kd = cell_pair(1)
    a_ss, b_ss = cell_pair(2)
    a_ts, b_ts = cell_pair(4)
    a_td, b_td = cell_pair(5)
    a_sub, b_sub = cell_pair(7)
    a_rev, b_rev = cell_pair(8)
    a_ctrl, b_ctrl = cell_pair(9)

    row.a_kd, row.b_kd = parse_int(a_kd), parse_int(b_kd)
    row.a_sig_str_lnd, row.a_sig_str_att = parse_x_of_y(a_ss)
    row.b_sig_str_lnd, row.b_sig_str_att = parse_x_of_y(b_ss)
    row.a_total_str_lnd, row.a_total_str_att = parse_x_of_y(a_ts)
    row.b_total_str_lnd, row.b_total_str_att = parse_x_of_y(b_ts)
    row.a_td_lnd, row.a_td_att = parse_x_of_y(a_td)
    row.b_td_lnd, row.b_td_att = parse_x_of_y(b_td)
    row.a_sub_att, row.b_sub_att = parse_int(a_sub), parse_int(b_sub)
    row.a_rev, row.b_rev = parse_int(a_rev), parse_int(b_rev)
    row.a_ctrl_sec, row.b_ctrl_sec = parse_ctrl_time(a_ctrl), parse_ctrl_time(b_ctrl)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="../data/ufc_fights_raw.csv")
    ap.add_argument("--years", type=int, default=10, help="How many years back to scrape")
    ap.add_argument("--limit-events", type=int, default=None, help="Cap events (for testing)")
    args = ap.parse_args()

    cutoff = (datetime.utcnow() - timedelta(days=365 * args.years)).date().isoformat()
    log.info("Scraping events on or after %s", cutoff)

    fetcher = Fetcher(delay=1.0)
    events = list_events(fetcher, since_date=cutoff)
    log.info("Found %d events in window", len(events))
    if args.limit_events:
        events = events[: args.limit_events]

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FightRow.__dataclass_fields__.keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        total = 0
        for ev in events:
            try:
                fight_urls = list_fights_for_event(fetcher, ev["url"])
            except Exception as e:
                log.error("Skipping event %s: %s", ev["name"], e)
                continue
            for fu in fight_urls:
                try:
                    row = parse_fight(fetcher, fu, ev)
                    if row:
                        writer.writerow(asdict(row))
                        total += 1
                except Exception as e:
                    log.error("Fight %s failed: %s", fu, e)
            log.info("Event '%s' (%s) processed - running total %d fights",
                     ev["name"], ev["date"], total)

    log.info("Done. Wrote %d fights -> %s", total, out_path)


if __name__ == "__main__":
    main()
