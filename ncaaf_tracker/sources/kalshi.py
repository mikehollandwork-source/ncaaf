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


def list_markets(session, series: str, status: str | None = "open", min_close_ts: int | None = None,
                 max_close_ts: int | None = None) -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "limit": 200}
        if status:
            params["status"] = status
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
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
        "open_time": m.get("open_time"),
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
    slack = timedelta(0) if row.get("kickoff_source") == "espn" else timedelta(minutes=30)
    if ko <= now - slack:
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


def fetch_trades(session, ticker: str, since: datetime, until: datetime | None = None,
                 max_pages: int = 50) -> list[dict]:
    """Every trade in [since, until] (Kalshi keeps full history), newest first."""
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"ticker": ticker, "min_ts": int(since.timestamp()), "limit": 1000}
        if until is not None:
            params["max_ts"] = int(until.timestamp())
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


def backfill_trades(session, now: datetime, since: datetime, kickoff_fn, meta: dict,
                    until: datetime | None = None) -> dict[str, list[dict]]:
    """Full pre-kickoff trade history for every game-winner market that closed
    in [since, until]: the market rows plus every trade up to ESPN's kickoff."""
    snap = iso(now)
    raw = []
    for status in ("settled", "closed"):
        try:
            raw += list_markets(session, GAME_SERIES, status=status, min_close_ts=int(since.timestamp()),
                                max_close_ts=int(until.timestamp()) if until else None)
        except Exception as e:
            log.warning("kalshi list %s markets failed: %s", status, e)
    rows = {m["ticker"]: market_row(m, GAME_SERIES, snap) for m in raw if m.get("ticker")}
    rows = list(rows.values())
    meta["kalshi_backfill_markets"] = len(rows)
    meta["kalshi_backfill_espn_kickoffs"] = apply_espn_kickoffs(rows, kickoff_fn)
    trades, failures, done = [], 0, 0
    for r in rows:
        ko = parse_time(r.get("kickoff_est"))
        if r.get("kickoff_source") != "espn" or ko is None:
            continue
        if failures >= 10:
            log.error("kalshi trade backfill: repeated failures, stopping")
            break
        try:
            opened = parse_time(r.get("open_time")) or ko - timedelta(days=21)
            trades += fetch_trades(session, r["ticker"], min(opened, ko - timedelta(hours=1)), ko, max_pages=200)
            failures, done = 0, done + 1
        except Exception as e:
            failures += 1
            log.warning("kalshi trades %s failed: %s", r["ticker"], e)
    for t in trades:
        t["snapshot_ts"] = snap
    meta["kalshi_backfill_tickers"] = done
    meta["rows_kalshi_backfill_trades"] = len(trades)
    return {"kalshi_markets": rows, "kalshi_trades": trades}


def game_key(event_ticker: str | None) -> str | None:
    """KXNCAAFSPREAD-26OCT03ARKTXAM -> 26OCT03ARKTXAM (shared by every series for one game)."""
    return event_ticker.split("-", 1)[1] if event_ticker and "-" in event_ticker else None


def apply_espn_kickoffs(rows: list[dict], kickoff_fn) -> int:
    """Replace the expiration-based estimate with ESPN's kickoff where the game matches.
    Kalshi's expiration offset isn't reliable for late games, and a wrong
    estimate means storing in-game trades as if they were pregame."""
    if kickoff_fn is None:
        return 0
    teams: dict[str, list] = {}
    for r in rows:
        if r["series"] == GAME_SERIES and r.get("yes_sub_title"):
            teams.setdefault(game_key(r["event_ticker"]), []).append(r)
    ko = {}
    for key, rs in teams.items():
        names = sorted({r["yes_sub_title"] for r in rs})
        if len(names) == 2:
            k = kickoff_fn(names[0], names[1], parse_time(rs[0].get("kickoff_est")))
            if k is not None:
                ko[key] = k
    n = 0
    for r in rows:
        k = ko.get(game_key(r["event_ticker"]))
        r["kickoff_source"] = "espn" if k else "estimate"
        if k:
            r["kickoff_est"] = iso(k)
            n += 1
    return n


def collect(session, now: datetime, horizon: datetime | None, meta: dict,
            kickoff_fn=None, trades_since: datetime | None = None) -> dict[str, list[dict]]:
    snap = iso(now)
    series_list = [s.strip() for s in env("KALSHI_SERIES", DEFAULT_SERIES).split(",") if s.strip()]
    depth = int(env("KALSHI_BOOK_DEPTH", "10"))
    book_series = set(env("KALSHI_BOOK_SERIES", GAME_SERIES).split(","))
    lookback = timedelta(minutes=int(env("TRADE_LOOKBACK_MIN", "75")))
    # GitHub runs the collector only every few hours: fetch every trade since the last save
    # (15 min overlap, deduped by trade_id in analysis), capped at a week.
    since_all = max(trades_since - timedelta(minutes=15), now - timedelta(days=7)) if trades_since else now - lookback

    all_rows = []
    for series in series_list:
        try:
            raw = list_markets(session, series)
        except Exception as e:
            log.warning("kalshi list %s failed: %s", series, e)
            meta[f"kalshi_{series}_error"] = str(e)[:200]
            continue
        all_rows += [market_row(m, series, snap) for m in raw]
    meta["kalshi_espn_kickoffs"] = apply_espn_kickoffs(all_rows, kickoff_fn)

    markets, books, trades = [], [], []
    for series in series_list:
        rows = [r for r in all_rows if r["series"] == series and in_window(r, now, horizon)]
        meta[f"kalshi_{series}_markets"] = len(rows)
        if not rows:
            log.warning("kalshi series %s returned no in-window markets", series)
        markets.extend(rows)
        if series not in book_series:
            continue
        failures = 0
        for r in rows:
            if failures >= 6:
                log.error("kalshi: repeated book/trade failures, skipping the rest this run")
                break
            t = r["ticker"]
            ko = parse_time(r.get("kickoff_est"))
            try:
                ob = get_json(session, f"{BASE}/markets/{t}/orderbook", {"depth": depth},
                              min_interval=PAUSE)
                books.extend(parse_orderbook(ob, t, snap, depth))
                failures = 0
            except Exception as e:
                failures += 1
                log.warning("kalshi book %s failed: %s", t, e)
            try:
                got = fetch_trades(session, t, since_all, ko if r.get("kickoff_source") == "espn" else None)
                if ko is not None and r.get("kickoff_source") == "espn":
                    got = [x for x in got if (parse_time(x["created_time"]) or now) < ko]
                trades.extend(got)
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
