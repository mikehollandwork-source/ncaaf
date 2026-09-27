"""Polymarket public data (no auth).

  Gamma API  -> event/market discovery, prices, cumulative volume   (poly_markets)
  CLOB API   -> order book depth per outcome token                   (poly_books)
  Data API   -> individual trades with side (BUY/SELL) and outcome   (poly_trades)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..util import env, get_json, iso, json_list, log, parse_time, to_float

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
PAUSE = 0.1


def _cfb_series_ids(session) -> list[str]:
    override = env("POLY_SERIES_IDS")
    if override:
        return [s.strip() for s in override.split(",") if s.strip()]
    try:
        sports = get_json(session, f"{GAMMA}/sports")
    except Exception as e:
        log.warning("polymarket /sports failed: %s", e)
        return []
    ids = []
    for s in sports or []:
        code = str(s.get("sport", "")).lower()
        if code in ("cfb", "ncaaf", "college-football"):
            for sid in str(s.get("series", "")).split(","):
                if sid.strip():
                    ids.append(sid.strip())
    return ids


def list_events(session) -> list[dict]:
    events: dict[str, dict] = {}
    queries = [{"series_id": sid} for sid in _cfb_series_ids(session)]
    queries += [{"tag_slug": env("POLY_TAG_SLUG", "cfb")}]
    for q in queries:
        offset = 0
        while offset < 5000:
            params = {**q, "closed": "false", "limit": 100, "offset": offset}
            try:
                page = get_json(session, f"{GAMMA}/events", params, min_interval=PAUSE)
            except Exception as e:
                log.warning("polymarket events %s failed: %s", q, e)
                break
            if not page:
                break
            for ev in page:
                events[str(ev.get("id"))] = ev
            if len(page) < 100:
                break
            offset += 100
    return list(events.values())


def market_kickoff(ev: dict, m: dict) -> datetime | None:
    return (parse_time(m.get("gameStartTime")) or parse_time(ev.get("startTime"))
            or parse_time(m.get("eventStartTime")))


def market_rows(events: list[dict], now: datetime, horizon: datetime | None) -> list[dict]:
    snap = iso(now)
    # Props (player/team yards, exact margin, halves) are ~2/3 of CFB markets; skip them.
    keep = {t.strip() for t in env("POLY_MARKET_TYPES", "moneyline,spreads,totals").split(",") if t.strip()}
    rows = []
    for ev in events:
        for m in ev.get("markets", []) or []:
            if m.get("closed"):
                continue
            mtype = (m.get("sportsMarketType") or "").lower()
            if keep and mtype and mtype not in keep:
                continue
            ko = market_kickoff(ev, m)
            # No game start -> futures (e.g. national title); not a single game.
            if ko is None or ko < now - timedelta(hours=1):
                continue
            if horizon is not None and ko > horizon:
                continue
            outcomes = json_list(m.get("outcomes"))
            prices = json_list(m.get("outcomePrices"))
            tokens = json_list(m.get("clobTokenIds"))
            row = {
                "snapshot_ts": snap,
                "event_id": ev.get("id"),
                "event_slug": ev.get("slug"),
                "event_title": ev.get("title"),
                "market_id": m.get("id"),
                "condition_id": m.get("conditionId"),
                "question": m.get("question"),
                "market_slug": m.get("slug"),
                "sports_market_type": m.get("sportsMarketType"),
                "line": to_float(m.get("line")),
                "game_start_time": iso(ko),
                "best_bid": to_float(m.get("bestBid")),
                "best_ask": to_float(m.get("bestAsk")),
                "last_trade_price": to_float(m.get("lastTradePrice")),
                "volume": to_float(m.get("volumeNum") or m.get("volume")),
                "volume_24h": to_float(m.get("volume24hr")),
                "liquidity": to_float(m.get("liquidityNum") or m.get("liquidity")),
                "accepting_orders": m.get("acceptingOrders"),
            }
            for i in range(2):
                row[f"outcome_{i}"] = outcomes[i] if i < len(outcomes) else None
                row[f"price_{i}"] = to_float(prices[i]) if i < len(prices) else None
                row[f"token_{i}"] = str(tokens[i]) if i < len(tokens) else None
            rows.append(row)
    return rows


def is_moneyline(r: dict) -> bool:
    t = (r.get("sports_market_type") or "").lower()
    if t:
        return t == "moneyline"
    q = (r.get("question") or "").lower()
    return r.get("line") is None and "spread" not in q and "o/u" not in q and "total" not in q


def fetch_books(session, token_meta: dict[str, tuple[str, str]], snap: str, depth: int) -> list[dict]:
    rows = []
    ids = list(token_meta)
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        try:
            resp = session.post(f"{CLOB}/books", json=[{"token_id": t} for t in chunk], timeout=30)
            resp.raise_for_status()
            books = resp.json()
        except Exception as e:
            log.warning("polymarket books chunk failed: %s", e)
            continue
        for b in books or []:
            tok = str(b.get("asset_id"))
            cond, outcome = token_meta.get(tok, (b.get("market"), None))
            for side, key, desc in (("bid", "bids", True), ("ask", "asks", False)):
                lv = [(to_float(x.get("price")), to_float(x.get("size"))) for x in b.get(key, []) or []]
                lv = [x for x in lv if x[0] is not None and x[1] is not None]
                lv.sort(key=lambda x: -x[0] if desc else x[0])  # don't trust API ordering
                for rank, (p, q) in enumerate(lv[:depth]):
                    rows.append({"snapshot_ts": snap, "condition_id": cond, "token_id": tok,
                                 "outcome": outcome, "side": side, "level": rank,
                                 "price": p, "size": q})
    return rows


def fetch_trades(session, condition_id: str, since: datetime) -> list[dict]:
    data = get_json(session, f"{DATA}/trades",
                    {"market": condition_id, "limit": 500, "takerOnly": "true"},
                    min_interval=PAUSE)
    out = []
    for t in data or []:
        ts = parse_time(t.get("timestamp"))
        if ts is None or ts < since:
            continue
        out.append({
            "trade_id": f"{t.get('transactionHash')}:{t.get('asset')}:{t.get('side')}:{t.get('size')}",
            "condition_id": condition_id,
            "token_id": t.get("asset"),
            "outcome": t.get("outcome"),
            "side": t.get("side"),
            "price": to_float(t.get("price")),
            "size": to_float(t.get("size")),
            "trade_ts": iso(ts),
        })
    return out


def collect(session, now: datetime, horizon: datetime | None, meta: dict) -> dict[str, list[dict]]:
    snap = iso(now)
    events = list_events(session)
    markets = market_rows(events, now, horizon)
    meta["poly_events"] = len(events)
    meta["poly_markets"] = len(markets)
    if not markets:
        log.warning("polymarket: no in-window CFB game markets found (events=%d)", len(events))
    depth = int(env("POLY_BOOK_DEPTH", "10"))
    lookback = timedelta(minutes=int(env("TRADE_LOOKBACK_MIN", "75")))
    ml = [r for r in markets if is_moneyline(r)]
    token_meta = {}
    for r in ml:
        for i in range(2):
            if r.get(f"token_{i}"):
                token_meta[r[f"token_{i}"]] = (r["condition_id"], r.get(f"outcome_{i}"))
    books = fetch_books(session, token_meta, snap, depth)
    trades, failures = [], 0
    for r in ml:
        if not r.get("condition_id"):
            continue
        if failures >= 6:
            log.error("polymarket: repeated trade failures, skipping the rest this run")
            break
        try:
            trades.extend(fetch_trades(session, r["condition_id"], now - lookback))
            failures = 0
        except Exception as e:
            failures += 1
            log.warning("polymarket trades %s failed: %s", r["condition_id"], e)
    for t in trades:
        t["snapshot_ts"] = snap
    return {"poly_markets": markets, "poly_books": books, "poly_trades": trades}
