"""Kalshi public market data (no auth needed for reads).

Captures, per pre-kickoff NCAAF market:
  * top-of-book / last / volume / open interest   (kalshi_markets)
  * order book depth, top N levels each side       (kalshi_books)  - moneyline only by default
  * individual trades with taker side              (kalshi_trades) - moneyline only by default
Trades with taker side are what tell you which way money actually moved;
the resting book only shows intent.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from ..util import env, get_json, iso, log, parse_time, to_float

BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = "KXNCAAFGAME,KXNCAAFSPREAD,KXNCAAFTOTAL"
GAME_SERIES = "KXNCAAFGAME"
PAUSE = 0.07  # stay well under the basic-tier read limit
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _money(m: dict, key: str) -> float | None:
    """Prices come as '<key>_dollars' strings on newer API versions, cents otherwise."""
    v = m.get(f"{key}_dollars")
    if v not in (None, ""):
        return to_float(v)
    v = to_float(m.get(key))
    return None if v is None else v / 100.0


def _count(m: dict, key: str) -> float | None:
    v = m.get(f"{key}_fp")
    return to_float(v if v not in (None, "") else m.get(key))


def ticker_date(ticker: str | None) -> datetime | None:
    """KXNCAAFGAME-26SEP05ECUALA -> 2026-09-05 (UTC midnight)."""
    if not ticker:
        return None
    m = re.search(r"-(\d\d)([A-Z]{3})(\d\d)", ticker)
    if not m or m.group(2) not in MONTHS:
        return None
    try:
        return datetime(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)),
                        tzinfo=timezone.utc)
    except ValueError:
        return None


def kickoff_estimate(m: dict) -> datetime | None:
    for k in ("occurrence_datetime", "event_start_time", "game_start_time"):
        t = parse_time(m.get(k))
        if t:
            return t
    exp = parse_time(m.get("expected_expiration_time"))
    if exp:
        # Calibrated against ESPN kickoffs (94 games): expiration = kickoff + 3h (median).
        return exp - timedelta(hours=3)
    return None


def list_markets(session, series: str) -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = get_json(session, f"{BASE}/markets", params, min_interval=PAUSE)
        out.extend(data.get("markets", []) or [])
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            return out


def market_row(m: dict, series: str, snap: str) -> dict:
    ko = kickoff_estimate(m)
    return {
        "snapshot_ts": snap,
        "series": series,
        "event_ticker": m.get("event_ticker"),
        "ticker": m.get("ticker"),
        "title": m.get("title"),
        "subtitle": m.get("subtitle"),
        "yes_sub_title": m.get("yes_sub_title"),
        "no_sub_title": m.get("no_sub_title"),
        "status": m.get("status"),
        "kickoff_est": iso(ko),
        "ticker_date": iso(ticker_date(m.get("event_ticker") or m.get("ticker"))),
        "expected_expiration_time": m.get("expected_expiration_time"),
        "close_time": m.get("close_time"),
        "strike_type": m.get("strike_type"),
        "floor_strike": to_float(m.get("floor_strike")),
        "cap_strike": to_float(m.get("cap_strike")),
        "yes_bid": _money(m, "yes_bid"),
        "yes_ask": _money(m, "yes_ask"),
        "no_bid": _money(m, "no_bid"),
        "no_ask": _money(m, "no_ask"),
        "last_price": _money(m, "last_price"),
        "previous_price": _money(m, "previous_price"),
        "volume": _count(m, "volume"),
        "volume_24h": _count(m, "volume_24h"),
        "open_interest": _count(m, "open_interest"),
        "liquidity": _money(m, "liquidity"),
    }


def in_window(row: dict, now: datetime, horizon: datetime | None) -> bool:
    """Lenient pre-kickoff filter. Analysis applies the exact ESPN kickoff cutoff."""
    ko = parse_time(row.get("kickoff_est"))
    if ko is None:
        td = parse_time(row.get("ticker_date"))
        if td is None:
            return horizon is None
        ko = td + timedelta(hours=23, minutes=59)
    if ko < now - timedelta(minutes=30):
        return False
    return horizon is None or ko <= horizon


def parse_orderbook(data: dict, ticker: str, snap: str, depth: int) -> list[dict]:
    ob = data.get("orderbook_fp") or data.get("orderbook") or {}
    rows = []
    for side in ("yes", "no"):
        levels = ob.get(f"{side}_dollars")
        dollars = bool(levels)
        if not levels:
            levels = ob.get(side) or []
        parsed = []
        for lvl in levels:
            if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                continue
            p, q = to_float(lvl[0]), to_float(lvl[1])
            if p is None or q is None:
                continue
            parsed.append((p if dollars else p / 100.0, q))
        # Kalshi books list bids only; best bid = highest price.
        parsed.sort(key=lambda x: -x[0])
        for rank, (p, q) in enumerate(parsed[:depth]):
            rows.append({"snapshot_ts": snap, "ticker": ticker, "side": side,
                         "level": rank, "price": p, "size": q})
    return rows


def fetch_trades(session, ticker: str, since: datetime) -> list[dict]:
    out, cursor = [], None
    for _ in range(20):
        params = {"ticker": ticker, "min_ts": int(since.timestamp()), "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = get_json(session, f"{BASE}/markets/trades", params, min_interval=PAUSE)
        for t in data.get("trades", []) or []:
            out.append({
                "trade_id": t.get("trade_id"),
                "ticker": t.get("ticker") or ticker,
                "created_time": t.get("created_time"),
                "yes_price": _money(t, "yes_price"),
                "no_price": _money(t, "no_price"),
                "count": _count(t, "count"),
                "taker_side": t.get("taker_side"),
            })
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            break
    return out


def collect(session, now: datetime, horizon: datetime | None, meta: dict) -> dict[str, list[dict]]:
    snap = iso(now)
    series_list = [s.strip() for s in env("KALSHI_SERIES", DEFAULT_SERIES).split(",") if s.strip()]
    depth = int(env("KALSHI_BOOK_DEPTH", "10"))
    book_series = set(env("KALSHI_BOOK_SERIES", GAME_SERIES).split(","))
    lookback = timedelta(minutes=int(env("TRADE_LOOKBACK_MIN", "75")))

    markets, books, trades = [], [], []
    for series in series_list:
        try:
            raw = list_markets(session, series)
        except Exception as e:
            log.warning("kalshi list %s failed: %s", series, e)
            meta[f"kalshi_{series}_error"] = str(e)[:200]
            continue
        rows = [r for r in (market_row(m, series, snap) for m in raw) if in_window(r, now, horizon)]
        meta[f"kalshi_{series}_markets"] = len(rows)
        if not rows:
            log.warning("kalshi series %s returned no in-window markets (raw=%d)", series, len(raw))
        markets.extend(rows)
        if series not in book_series:
            continue
        failures = 0
        for r in rows:
            if failures >= 6:
                log.error("kalshi: repeated book/trade failures, skipping the rest this run")
                break
            t = r["ticker"]
            try:
                ob = get_json(session, f"{BASE}/markets/{t}/orderbook", {"depth": depth},
                              min_interval=PAUSE)
                books.extend(parse_orderbook(ob, t, snap, depth))
                failures = 0
            except Exception as e:
                failures += 1
                log.warning("kalshi book %s failed: %s", t, e)
            try:
                trades.extend(fetch_trades(session, t, now - lookback))
            except Exception as e:
                failures += 1
                log.warning("kalshi trades %s failed: %s", t, e)
    for t in trades:
        t["snapshot_ts"] = snap
    return {"kalshi_markets": markets, "kalshi_books": books, "kalshi_trades": trades}


def discover_series(session) -> list[dict]:
    """Print-helper: list every Kalshi series that looks like college football."""
    data = get_json(session, f"{BASE}/series", {"category": "Sports"})
    return [
        {"ticker": s.get("ticker"), "title": s.get("title")}
        for s in data.get("series", []) or []
        if "NCAAF" in (s.get("ticker") or "").upper()
        or "college football" in (s.get("title") or "").lower()
    ]
