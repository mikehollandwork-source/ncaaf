"""The Odds API (the-odds-api.com): every US sportsbook's moneyline / spread /
total for NCAAF in one request. Requires ODDS_API_KEY.

Credit cost per call = (#markets) x (#regions). Default 3 markets x 3 regions = 9.
"""
from __future__ import annotations

from datetime import datetime

from ..util import env, iso, log, parse_time, to_float

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf"
DEFAULT_REGIONS = "us,eu,us_ex"   # eu = Pinnacle (sharp reference); us_ex = Novig/ProphetX exchanges
DEFAULT_MARKETS = "h2h,spreads,totals"


def fetch_odds(session, now: datetime, meta: dict) -> list[dict]:
    key = env("ODDS_API_KEY")
    if not key:
        log.warning("ODDS_API_KEY not set; skipping sportsbook odds")
        meta["oddsapi_status"] = "no_key"
        return []
    regions = env("ODDS_API_REGIONS", DEFAULT_REGIONS)
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": env("ODDS_API_MARKETS", DEFAULT_MARKETS),
        "oddsFormat": "american",
        "dateFormat": "iso",
        # Only games that haven't kicked off: tracking stops at kickoff.
        "commenceTimeFrom": iso(now),
    }
    resp = session.get(f"{BASE}/odds", params=params, timeout=45)
    if resp.status_code == 422 and regions != "us":
        # Unknown region on this plan/account -> fall back to plain US books.
        log.warning("odds api rejected regions=%s (%s); retrying with us", regions, resp.text[:200])
        params["regions"] = "us"
        resp = session.get(f"{BASE}/odds", params=params, timeout=45)
    meta["oddsapi_requests_remaining"] = resp.headers.get("x-requests-remaining")
    meta["oddsapi_requests_used"] = resp.headers.get("x-requests-used")
    meta["oddsapi_last_cost"] = resp.headers.get("x-requests-last")
    resp.raise_for_status()
    meta["oddsapi_status"] = "ok"
    return parse_odds(resp.json(), now)


def parse_odds(events: list, now: datetime) -> list[dict]:
    rows = []
    snap = iso(now)
    for ev in events or []:
        commence = parse_time(ev.get("commence_time"))
        if commence is None or commence <= now:
            continue
        for bk in ev.get("bookmakers", []) or []:
            for mk in bk.get("markets", []) or []:
                for oc in mk.get("outcomes", []) or []:
                    rows.append({
                        "snapshot_ts": snap,
                        "oddsapi_id": ev.get("id"),
                        "commence_time": iso(commence),
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "book": bk.get("key"),
                        "book_last_update": bk.get("last_update"),
                        "market": mk.get("key"),
                        "market_last_update": mk.get("last_update"),
                        "outcome": oc.get("name"),
                        "price": to_float(oc.get("price")),
                        "point": to_float(oc.get("point")),
                    })
    return rows
