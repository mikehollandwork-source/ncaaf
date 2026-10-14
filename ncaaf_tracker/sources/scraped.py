"""Keyless sportsbook lines from public site feeds (what the web pages load).

Action Network  api.actionnetwork.com/web/v2/scoreboard/ncaaf
    DraftKings / FanDuel / BetMGM / Caesars / BetRivers / bet365, the market
    OPENING line (book 30) and public betting splits (% of tickets and % of
    money on each side).
Pinnacle        guest.api.arcadia.pinnacle.com  (the pinnacle.com web feed)
    The sharp reference line.

Rows use the same shape as the Odds API `odds` table (event_id, home/away,
book, market in h2h/spreads/totals, outcome, price, point) so analysis treats
every source the same way.
"""
from __future__ import annotations

from datetime import datetime

from ..util import env, get_json, iso, log, parse_time, to_float

AN = "https://api.actionnetwork.com/web"
AN_BOOKS = {15: "an_consensus", 30: "an_open", 68: "draftkings", 69: "fanduel", 71: "betrivers",
            75: "betmgm", 79: "bet365", 123: "caesars", 972: "betrivers_ny"}
AN_MARKETS = {"moneyline": "h2h", "spread": "spreads", "total": "totals"}

PIN = "https://guest.api.arcadia.pinnacle.com/0.1"
PIN_WEB_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"  # public key the pinnacle.com web app sends
PIN_LEAGUES = {880: "FBS", 202331: "FCS"}


# ------------------------------------------------------------------ Action Network

def an_fetch(session, now: datetime, meta: dict) -> tuple[list[dict], list[dict]]:
    books = ",".join(str(b) for b in AN_BOOKS)
    games: dict = {}
    weeks = env("AN_WEEKS")  # optional override, e.g. "6,7"
    queries = []
    for div in ("FBS", "FCS"):
        if weeks:
            queries += [{"bookIds": books, "division": div, "week": w, "season": now.year} for w in weeks.split(",")]
        else:
            # current + next week (lines post ~a week ahead; lookahead lines earlier)
            queries += [{"bookIds": books, "division": div, "week": w, "season": now.year}
                        for w in _an_weeks(session, now)]
    for q in queries:
        try:
            data = get_json(session, f"{AN}/v2/scoreboard/ncaaf", q, min_interval=0.3)
        except Exception as e:
            log.warning("action network %s failed: %s", q, e)
            continue
        for g in data.get("games", []) or []:
            games[g.get("id")] = g
    meta["an_games"] = len(games)
    odds, splits = [], []
    for g in games.values():
        o, s = an_parse_game(g, now)
        odds += o
        splits += s
    return odds, splits


def an_current_week(session) -> int | None:
    """The scoreboard without a week returns the current (most recent) week."""
    try:
        data = get_json(session, f"{AN}/v2/scoreboard/ncaaf", {"bookIds": "15", "division": "FBS"})
        weeks = [g.get("week") for g in data.get("games", []) if g.get("week")]
        return max(set(weeks), key=weeks.count) if weeks else None
    except Exception as e:
        log.warning("action network week discovery failed: %s", e)
        return None


def _an_weeks(session, now: datetime) -> list[int]:
    cur = an_current_week(session)
    # current + next two weeks: books post next week's lines Sunday, lookahead lines earlier
    return [] if cur is None else [cur, cur + 1, cur + 2]


def an_week_games(session, season: int, week: int) -> list[dict]:
    games = []
    for div in ("FBS", "FCS"):
        try:
            data = get_json(session, f"{AN}/v2/scoreboard/ncaaf",
                            {"bookIds": "15", "division": div, "week": week, "season": season}, min_interval=0.3)
            games += data.get("games", []) or []
        except Exception as e:
            log.warning("action network week %s %s failed: %s", week, div, e)
    return games


def an_history(session, game: dict, now: datetime) -> list[dict]:
    """Every line change per book, opener to kickoff, for one game."""
    books = ",".join(str(b) for b in AN_BOOKS if b != 15)
    data = get_json(session, f"{AN}/v2/markets/event/{game['id']}/history", {"bookIds": books},
                    min_interval=0.3)
    teams = {t.get("id"): t for t in game.get("teams", []) or []}
    home, away = teams.get(game.get("home_team_id")), teams.get(game.get("away_team_id"))
    if not home or not away:
        return []
    start = parse_time(game.get("start_time"))
    base = {"fetched_ts": iso(now), "source": "an_history", "event_id": f"an:{game['id']}",
            "commence_time": iso(start), "home_team": home.get("full_name"), "away_team": away.get("full_name")}
    name = {"home": base["home_team"], "away": base["away_team"], "over": "Over", "under": "Under"}
    rows = []
    for book_id, mk in (data or {}).items():
        try:
            book = AN_BOOKS.get(int(book_id), f"an_{book_id}")
        except ValueError:
            continue
        for an_type, market in AN_MARKETS.items():
            for o in (mk.get("event") or {}).get(an_type, []) or []:
                for h in o.get("history", []) or []:
                    rows.append({**base, "book": book, "market": market,
                                 "outcome": name.get(o.get("side"), o.get("side")),
                                 "price": to_float(h.get("odds")),
                                 "point": None if market == "h2h" else to_float(h.get("value")),
                                 "updated_at": h.get("updated_at"), "line_status": h.get("line_status")})
    return rows


def an_history_for(session, now: datetime, weeks: list[int], started_after: datetime | None,
                   started_before: datetime, meta: dict) -> list[dict]:
    rows, n, failures = [], 0, 0
    for w in weeks:
        for g in an_week_games(session, now.year, w):
            st = parse_time(g.get("start_time"))
            if st is None or st > started_before or (started_after and st < started_after):
                continue
            if failures >= 8:
                log.error("action network history: repeated failures, stopping")
                meta["an_history_games"] = n
                return rows
            try:
                rows += an_history(session, g, now)
                n += 1
                failures = 0
            except Exception as e:
                failures += 1
                log.warning("action network history %s failed: %s", g.get("id"), e)
    meta["an_history_games"] = n
    return rows


def an_parse_game(g: dict, now: datetime) -> tuple[list[dict], list[dict]]:
    start = parse_time(g.get("start_time"))
    if start is None or start <= now or g.get("status") not in (None, "scheduled"):
        return [], []
    teams = {t.get("id"): t for t in g.get("teams", []) or []}
    home, away = teams.get(g.get("home_team_id")), teams.get(g.get("away_team_id"))
    if not home or not away:
        return [], []
    base = {
        "snapshot_ts": iso(now), "source": "actionnetwork", "event_id": f"an:{g.get('id')}",
        "commence_time": iso(start), "home_team": home.get("full_name"), "away_team": away.get("full_name"),
    }
    name = {"home": base["home_team"], "away": base["away_team"], "over": "Over", "under": "Under"}
    odds, splits = [], []
    for book_id, mk in (g.get("markets") or {}).items():
        book = AN_BOOKS.get(int(book_id), f"an_{book_id}")
        for an_type, market in AN_MARKETS.items():
            for o in (mk.get("event") or {}).get(an_type, []) or []:
                if o.get("is_live"):
                    continue
                side = o.get("side")
                odds.append({**base, "book": book, "market": market, "outcome": name.get(side, side),
                             "price": to_float(o.get("odds")),
                             "point": None if market == "h2h" else to_float(o.get("value")),
                             "line_status": o.get("line_status")})
                bi = o.get("bet_info") or {}
                if int(book_id) == 15 and bi:
                    splits.append({**base, "market": market, "side": side,
                                   "tickets_pct": to_float((bi.get("tickets") or {}).get("percent")),
                                   "money_pct": to_float((bi.get("money") or {}).get("percent"))})
    return odds, splits


def an_props(session, game: dict, now: datetime) -> list[dict]:
    """Player + team props for one game: every book's line and odds, one row per side."""
    books = ",".join(str(b) for b in AN_BOOKS if b != 30)
    data = get_json(session, f"{AN}/v2/games/{game['id']}/props", {"bookIds": books}, min_interval=0.3)
    return an_parse_props(data, game, now)


def an_parse_props(data: dict, game: dict, now: datetime) -> list[dict]:
    teams = {t.get("id"): t for t in game.get("teams", []) or []}
    side_of = {game.get("home_team_id"): "home", game.get("away_team_id"): "away"}
    home, away = teams.get(game.get("home_team_id"), {}), teams.get(game.get("away_team_id"), {})
    players = data.get("players") or {}
    base = {"snapshot_ts": iso(now), "event_id": f"an:{game.get('id')}",
            "commence_time": iso(parse_time(game.get("start_time"))),
            "home_team": home.get("full_name"), "away_team": away.get("full_name")}
    rows = []
    for group in ("player_props", "game_props"):
        for ptype, markets in (data.get(group) or {}).items():
            for m in markets or []:
                pl = players.get(str(m.get("player_id"))) or {}
                team_id = m.get("team_id") or pl.get("team_id")
                for book_id, outs in (m.get("lines") or {}).items():
                    try:
                        book = AN_BOOKS.get(int(book_id), f"an_{book_id}")
                    except ValueError:
                        continue
                    for o in outs or []:
                        if o.get("is_live"):
                            continue
                        tid = team_id or o.get("team_id")   # team totals carry it per outcome
                        rows.append({
                            **base, "prop_group": group, "prop_type": ptype,
                            "prop_name": m.get("custom_pick_type_name"),
                            "milestone": "_milestones_" in ptype,
                            "player_id": m.get("player_id"), "player": pl.get("full_name"),
                            "player_info": pl.get("display_text"),         # e.g. "FSU - QB"
                            "team_id": tid, "team": (teams.get(tid) or {}).get("full_name"),
                            "home_away": side_of.get(tid),
                            "book": book, "side": o.get("side"), "line": to_float(o.get("value")),
                            "odds": to_float(o.get("odds")), "alt": o.get("is_alt_market"),
                            "line_status": o.get("line_status"),
                        })
    return rows


def an_props_for(session, now: datetime, start: datetime, end: datetime, meta: dict) -> list[dict]:
    """Props for every game kicking off in [start, end]."""
    cur = an_current_week(session)
    meta["an_props_week"] = cur
    if cur is None:
        return []
    rows, n, failures = [], 0, 0
    seen, statuses = 0, {}
    for w in (cur, cur + 1):
        for g in an_week_games(session, now.year, w):
            seen += 1
            st = parse_time(g.get("start_time"))
            if st is None or not (start < st <= end):
                continue
            statuses[g.get("status")] = statuses.get(g.get("status"), 0) + 1
            if g.get("status") not in (None, "scheduled"):
                continue
            meta["an_props_candidates"] = meta.get("an_props_candidates", 0) + 1
            if failures >= 6:
                log.error("action network props: repeated failures, stopping")
                meta["an_props_games"] = n
                return rows
            try:
                got = an_props(session, g, now)
                failures = 0
            except Exception as e:
                failures += 1
                log.warning("action network props %s failed: %s", g.get("id"), e)
                continue
            rows += got
            n += bool(got)
    meta["an_props_games"] = n
    meta["an_props_scanned"] = seen
    meta["an_props_window_statuses"] = ",".join(f"{k}:{v}" for k, v in statuses.items())
    return rows


def an_final_lines(session, season: int, weeks=range(0, 21)) -> list[dict]:
    """Final (closing) lines per book plus the market opener (book 30) for every
    finished game of a season, from the past-week scoreboards."""
    books = ",".join(str(b) for b in (15, 30, 68, 69))
    rows = []
    for w in weeks:
        for div in ("FBS", "FCS"):
            try:
                data = get_json(session, f"{AN}/v2/scoreboard/ncaaf",
                                {"bookIds": books, "division": div, "week": w, "season": season}, min_interval=0.3)
            except Exception as e:
                log.warning("action network %s week %s %s failed: %s", season, w, div, e)
                continue
            for g in data.get("games", []) or []:
                if g.get("status") != "complete":
                    continue
                teams = {t.get("id"): t for t in g.get("teams", []) or []}
                home, away = teams.get(g.get("home_team_id"), {}), teams.get(g.get("away_team_id"), {})
                name = {"home": home.get("full_name"), "away": away.get("full_name"), "over": "Over", "under": "Under"}
                for book_id, mk in (g.get("markets") or {}).items():
                    book = AN_BOOKS.get(int(book_id), f"an_{book_id}")
                    for an_type, market in AN_MARKETS.items():
                        for o in (mk.get("event") or {}).get(an_type, []) or []:
                            bi = o.get("bet_info") or {}
                            rows.append({
                                "season": season, "week": g.get("week"), "event_id": f"an:{g.get('id')}",
                                "commence_time": iso(parse_time(g.get("start_time"))),
                                "home_team": name["home"], "away_team": name["away"],
                                "book": book, "market": market, "side": o.get("side"),
                                "outcome": name.get(o.get("side"), o.get("side")),
                                "price": to_float(o.get("odds")),
                                "point": None if market == "h2h" else to_float(o.get("value")),
                                "money_pct": to_float((bi.get("money") or {}).get("percent")),
                                "tickets_pct": to_float((bi.get("tickets") or {}).get("percent")),
                            })
    return rows


# ------------------------------------------------------------------ Pinnacle

def pin_fetch(session, now: datetime, meta: dict) -> list[dict]:
    headers = {"x-api-key": env("PINNACLE_WEB_KEY", PIN_WEB_KEY), "Referer": "https://www.pinnacle.com/"}
    rows = []
    for league in PIN_LEAGUES:
        try:
            matchups = session.get(f"{PIN}/leagues/{league}/matchups", headers=headers, timeout=25)
            matchups.raise_for_status()
            markets = session.get(f"{PIN}/leagues/{league}/markets/straight", headers=headers, timeout=25)
            markets.raise_for_status()
        except Exception as e:
            log.warning("pinnacle league %s failed: %s", league, e)
            continue
        rows += pin_parse(matchups.json(), markets.json(), now)
    meta["pinnacle_rows"] = len(rows)
    return rows


def pin_parse(matchups: list, markets: list, now: datetime) -> list[dict]:
    games = {}
    for m in matchups or []:
        if m.get("parentId") or m.get("isLive"):
            continue
        parts = {p.get("alignment"): p for p in m.get("participants", []) or []}
        if "home" not in parts or "away" not in parts:
            continue  # futures / props have 'neutral' participants
        start = parse_time(m.get("startTime"))
        if start is None or start <= now:
            continue
        games[m["id"]] = {
            "snapshot_ts": iso(now), "source": "pinnacle", "event_id": f"pin:{m['id']}",
            "commence_time": iso(start), "home_team": parts["home"].get("name"),
            "away_team": parts["away"].get("name"),
            "_pid": {parts["home"].get("id"): "home", parts["away"].get("id"): "away"},
        }
    rows = []
    for k in markets or []:
        g = games.get(k.get("matchupId"))
        if g is None or k.get("period") != 0 or k.get("isAlternate"):
            continue
        market = {"moneyline": "h2h", "spread": "spreads", "total": "totals"}.get(k.get("type"))
        if market is None:
            continue
        base = {x: v for x, v in g.items() if not x.startswith("_")}
        for p in k.get("prices", []) or []:
            side = p.get("designation") or g["_pid"].get(p.get("participantId"))
            outcome = {"home": g["home_team"], "away": g["away_team"], "over": "Over", "under": "Under"}.get(side)
            if outcome is None:
                continue
            rows.append({**base, "book": "pinnacle", "market": market, "outcome": outcome,
                         "price": to_float(p.get("price")),
                         "point": None if market == "h2h" else to_float(p.get("points"))})
    return rows
