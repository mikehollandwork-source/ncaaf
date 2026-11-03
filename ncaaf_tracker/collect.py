"""One collection pass. Run on a schedule (see .github/workflows/collect.yml).

Modes
  full     every source, every pre-kickoff game (the hourly snapshot)
  closing  only games kicking off within CLOSING_WINDOW_MIN; exits without
           writing if there are none. Tightens the closing-line snapshot.
  auto     full if minute < 15, else closing (default)
  backfill ESPN schedule/results/open-close lines from --since to today, plus
           Action Network's full line history (every book's line changes,
           opener to kickoff) for every game already played this season

Sportsbook lines need no API key: Action Network's public feed (DraftKings,
FanDuel, BetMGM, Caesars, BetRivers, bet365, the market opener, public
money/ticket splits) and Pinnacle's web feed. The Odds API is used only if
ODDS_API_KEY is set. The exact line history for games that have kicked off is
saved by the first full run after the previous save is 20h+ old.

Tracking stops at kickoff: every source filters to games that haven't started
(prediction markets with a +/-1h lenient window because their kickoff time is
estimated; analysis then trims to the exact ESPN kickoff).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .sources import dkn, espn, kalshi, oddsapi, polymarket, scraped, vsin
from .util import env, iso, log, make_session, parse_time, to_float, utcnow, write_table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("auto", "full", "closing", "backfill", "history", "trades"), default="auto")
    ap.add_argument("--data-dir", default=env("DATA_DIR", "data"))
    ap.add_argument("--sources", default="espn,an,dkn,pinnacle,odds,kalshi,poly",
                    help="comma list of sources to poll")
    ap.add_argument("--since", help="backfill/history start date YYYY-MM-DD")
    ap.add_argument("--until", help="history end date YYYY-MM-DD (default today)")
    ap.add_argument("--probe", action="store_true",
                    help="hit one endpoint per source, print HTTP status, and exit")
    ap.add_argument("--discover", action="store_true",
                    help="print Kalshi CFB series + Polymarket CFB series ids and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session = make_session()
    if args.discover:
        print(json.dumps({"kalshi_series": kalshi.discover_series(session),
                          "polymarket_series_ids": polymarket._cfb_series_ids(session)}, indent=2))
        return 0

    if args.probe:
        return probe(session, Path(args.data_dir).parent / "probe" if args.data_dir else None)

    now = utcnow()
    data_dir = Path(args.data_dir)
    sources = set(args.sources.split(","))
    mode = args.mode
    if mode == "auto":
        mode = "full" if now.minute < 15 else "closing"
    meta: dict = {"run_ts": iso(now), "mode": mode}
    tables: dict[str, list[dict]] = {}
    window = timedelta(minutes=int(env("CLOSING_WINDOW_MIN", "100")))

    # ---- master schedule / results
    if mode == "trades":
        # Full pre-kickoff Kalshi trade history for every finished game in [since, until].
        since = parse_time(f"{args.since}T00:00:00Z")
        until = parse_time(f"{args.until}T00:00:00Z") if args.until else now
        games = espn.fetch_games(session, since - timedelta(days=1), until + timedelta(days=1))
        # markets settled before Kalshi's historical cutoff live in its archive endpoints
        historical = until < parse_time("2026-08-14T00:00:00Z")
        meta["kalshi_historical"] = historical
        got = kalshi.backfill_trades(session, now, since, espn_kickoff_lookup(games), meta, until, historical)
        for name, rows in got.items():
            write_table(data_dir, name, rows, now)
        write_table(data_dir, "runs", [meta], now)
        log.info("kalshi trades backfill: %s", {k: v for k, v in meta.items() if "kalshi" in k})
        return 0
    if mode == "history":
        # ESPN only, any date range (e.g. a past season): games, ranks, results, and one
        # game-page load per finished game for its closing odds, box score and plays.
        since = parse_time(f"{args.since}T00:00:00Z")
        until = parse_time(f"{args.until}T00:00:00Z") if args.until else now
        games = espn.fetch_games(session, since, until)
        save_box_scores(session, data_dir, games, now, meta)
        for g in games:
            g["snapshot_ts"] = iso(now)
        write_table(data_dir, "espn_games", games, now)
        write_table(data_dir, "results", espn.results(games), now)
        lines, lines_1h = [], []   # one file per table for all seasons (same-second writes share a name)
        for season in sorted({since.year if since.month >= 7 else since.year - 1,
                              until.year if until.month >= 7 else until.year - 1}):
            got = scraped.an_final_lines(session, season)
            meta[f"an_final_lines_{season}"] = len(got)
            lines += got
            got = scraped.an_final_lines(session, season, period="firsthalf")
            meta[f"an_final_lines_1h_{season}"] = len(got)
            lines_1h += got
        write_table(data_dir, "an_final_lines", lines, now)
        write_table(data_dir, "an_final_lines_1h", lines_1h, now)
        meta.update({"history_since": args.since, "history_until": args.until, "espn_games": len(games)})
        write_table(data_dir, "runs", [meta], now)
        log.info("history %s..%s: %d games, %s game pages", args.since, args.until, len(games), meta.get("box_scores"))
        return 0
    if mode == "backfill":
        since = parse_time(f"{args.since}T00:00:00Z") if args.since else now - timedelta(days=45)
        games = espn.fetch_games(session, since, now + timedelta(days=14))
        log.info("filled ESPN open/close for %d finished games", espn.fill_closed_odds(session, games))
        for g in games:
            g["snapshot_ts"] = iso(now)
        write_table(data_dir, "espn_games", games, now)
        write_table(data_dir, "results", espn.results(games), now)
        save_box_scores(session, data_dir, games, now, meta)
        log.info("backfilled %d games since %s", len(games), iso(since))
        cur = scraped.an_current_week(session)
        if cur is not None:
            hist = scraped.an_history_for(session, now, list(range(0, cur + 1)), since, now, meta)
            write_table(data_dir, "an_history", hist, now)
            log.info("backfilled Action Network line history: %d rows, %s games",
                     len(hist), meta.get("an_history_games"))
        write_table(data_dir, "runs", [meta], now)
        return 0
    if mode == "full":
        games = espn.fetch_games(session, now - timedelta(days=3), now + timedelta(days=14))
        for g in games:
            g["snapshot_ts"] = iso(now)
        tables["espn_games"] = games
        tables["results"] = espn.results(games)   # finals from the last 3 days
        save_box_scores(session, data_dir, games, now, meta)
        horizon = None
    else:
        games = espn.fetch_games(session, now - timedelta(hours=12), now + timedelta(days=1))
        soon = [g for g in games
                if (k := parse_time(g.get("kickoff"))) and now < k <= now + window]
        meta["games_near_kickoff"] = len(soon)
        if not soon:
            log.info("closing mode: no kickoffs within %s; nothing to do", window)
            return 0
        horizon = now + window
        log.info("closing mode: %d games kick off within %s", len(soon), window)
    meta["espn_games"] = len(games)
    kickoff_fn = espn_kickoff_lookup(games)

    # ---- sportsbooks (keyless site feeds)
    in_h = (lambda r: parse_time(r["commence_time"]) <= horizon) if horizon else (lambda r: True)
    if "an" in sources:
        try:
            fh: list = []
            odds, splits = scraped.an_fetch(session, now, meta, first_half=fh)
            tables["an_odds"] = [r for r in odds if in_h(r)]
            tables["an_splits"] = [r for r in splits if in_h(r)]
            tables["an_odds_1h"] = [r for r in fh if in_h(r)]
        except Exception as e:
            log.exception("action network failed")
            meta["an_error"] = str(e)[:300]
        # Line history: GitHub drops many scheduled runs, so don't depend on one hour of the
        # day. Any full run saves it once the last save is 20h+ old, covering every game
        # since that save (Action Network keeps only about a week of history).
        last = last_history_save(data_dir)
        if mode == "full" and (last is None or now - last >= timedelta(hours=20)):
            since = max(last - timedelta(hours=36), now - timedelta(days=7)) if last else now - timedelta(days=7)
            cur = scraped.an_current_week(session)
            if cur is not None:
                tables["an_history"] = scraped.an_history_for(
                    session, now, [cur - 2, cur - 1, cur], since, now - timedelta(minutes=30), meta)
        # Player/team props (~570 rows a game): every ~6h for games in the next 3 days,
        # plus every near-kickoff run for games about to start (closing prop lines).
        try:
            if horizon is not None:
                tables["an_props"] = scraped.an_props_for(session, now, now, horizon, meta)
            else:
                last_p = last_save(data_dir, "an_props")
                if last_p is None or now - last_p >= timedelta(hours=int(env("PROPS_EVERY_H", "6"))):
                    tables["an_props"] = scraped.an_props_for(session, now, now, now + timedelta(days=3), meta)
        except Exception as e:
            log.exception("action network props failed")
            meta["an_props_error"] = str(e)[:300]
    try:
        tables["rule_picks"] = rule_snapshot(games, tables.get("an_odds", []), tables.get("an_splits", []), now)
        meta["rule_fires"] = sum(r["fires"] for r in tables["rule_picks"])
    except Exception as e:
        log.exception("rule snapshot failed")
        meta["rule_error"] = str(e)[:300]
    if "dkn" in sources:
        # DraftKings' own handle % / bets % per selection, every game in the next 7 days
        try:
            tables["dk_splits"] = dkn.fetch(session, now, meta)
        except Exception as e:
            log.exception("dk network splits failed")
            meta["dkn_error"] = str(e)[:300]
    if "vsin" in sources:   # off by default: VSiN withholds the split tables from scripted requests
        # DraftKings + Circa handle/bets % (VSiN only lists the current day's games)
        try:
            rows, unparsed = vsin.fetch(session, now, meta)
            tables["vsin_splits"] = rows
            for src, page in unparsed.items():       # page had data we couldn't read: keep it
                import gzip
                d = data_dir / "vsin_raw" / now.strftime("%Y-%m-%d")
                d.mkdir(parents=True, exist_ok=True)
                with gzip.open(d / f"vsin_{src}_{now.strftime('%Y%m%dT%H%M%SZ')}.html.gz", "wt", encoding="utf-8") as f:
                    f.write(page)
        except Exception as e:
            log.exception("vsin failed")
            meta["vsin_error"] = str(e)[:300]
    try:
        tables["split_picks"] = split_rules_snapshot(games, tables.get("an_odds", []), tables.get("an_splits", []), now)
        meta["split_rule_fires"] = sum(r["fires"] for r in tables["split_picks"])
    except Exception as e:
        log.exception("split rule snapshot failed")
        meta["split_rule_error"] = str(e)[:300]
    if "pinnacle" in sources:
        try:
            pfh: list = []
            tables["pinnacle_odds"] = [r for r in scraped.pin_fetch(session, now, meta, first_half=pfh) if in_h(r)]
            tables["pinnacle_odds_1h"] = [r for r in pfh if in_h(r)]
        except Exception as e:
            log.exception("pinnacle failed")
            meta["pinnacle_error"] = str(e)[:300]
    if "odds" in sources and env("ODDS_API_KEY"):
        try:
            rows = oddsapi.fetch_odds(session, now, meta)
            if horizon is not None:
                rows = [r for r in rows if parse_time(r["commence_time"]) <= horizon]
            tables["odds"] = rows
        except Exception as e:
            log.exception("odds api failed")
            meta["oddsapi_error"] = str(e)[:300]

    # ---- prediction markets (horizon padded: their kickoff time is an estimate)
    pm_horizon = horizon + timedelta(hours=1) if horizon else None
    if "kalshi" in sources:
        try:
            tables.update(kalshi.collect(session, now, pm_horizon, meta, kickoff_fn,
                                         trades_since=last_save(data_dir, "kalshi_trades")))
        except Exception as e:
            log.exception("kalshi failed")
            meta["kalshi_error"] = str(e)[:300]
    if "poly" in sources:
        try:
            tables.update(polymarket.collect(session, now, pm_horizon, meta, kickoff_fn))
        except Exception as e:
            log.exception("polymarket failed")
            meta["poly_error"] = str(e)[:300]

    if tables.get("kalshi_books"):
        try:
            tables["book_picks"] = book_rule_snapshot(games, tables.get("kalshi_markets", []),
                                                      tables["kalshi_books"], tables.get("an_odds", []), now)
            meta["book_rule_fires"] = sum(r["fires"] for r in tables["book_picks"])
        except Exception as e:
            log.exception("book rule snapshot failed")
            meta["book_rule_error"] = str(e)[:300]

    for name, rows in tables.items():
        meta[f"rows_{name}"] = len(rows)
        path = write_table(data_dir, name, rows, now)
        if path:
            log.info("wrote %s (%d rows)", path, len(rows))
    write_table(data_dir, "runs", [meta], now)
    log.info("run meta: %s", json.dumps(meta))
    return 0


def saved_ids(data_dir: Path, table: str) -> set:
    import csv
    import gzip
    ids = set()
    for f in data_dir.glob(f"{table}/*/{table}_*.csv.gz"):
        with gzip.open(f, "rt", newline="") as fh:
            ids.update(r["espn_id"] for r in csv.DictReader(fh))
    return ids


def saved_box_score_ids(data_dir: Path) -> set:
    return saved_ids(data_dir, "box_team")


def save_box_scores(session, data_dir: Path, games: list[dict], now, meta: dict) -> None:
    """Final box scores (team + every player's stat line) and play-by-play, once per game."""
    try:
        team, players, plays = espn.fetch_box_scores(session, games, saved_box_score_ids(data_dir), meta,
                                                     done_plays=saved_ids(data_dir, "plays"))
    except Exception as e:
        log.exception("box scores failed")
        meta["box_error"] = str(e)[:300]
        return
    write_table(data_dir, "box_team", team, now)
    write_table(data_dir, "box_players", players, now)
    write_table(data_dir, "plays", plays, now)
    meta["rows_box_players"] = len(players)
    meta["rows_plays"] = len(plays)


def last_save(data_dir: Path, table: str):
    """Timestamp of the newest saved file of a table, from its name."""
    files = sorted(data_dir.glob(f"{table}/*/{table}_*.csv.gz"))
    if not files:
        return None
    return parse_time(datetime.strptime(files[-1].name[len(table) + 1:-len(".csv.gz")],
                                        "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))


def last_history_save(data_dir: Path):
    return last_save(data_dir, "an_history")


RULE = "ranked_rlm_v1"   # locked 2026-10-14; do not edit, add a new version instead


def rule_snapshot(games: list[dict], an_odds: list[dict], an_splits: list[dict], now) -> list[dict]:
    """Locked rule: ranked game (either team AP top 25); DraftKings spread has moved 1+ pt from
    the market opener toward a side; that side has under 50% of both tickets and money.
    Every ranked game is logged each run with whether the rule fires right now."""
    import pandas as pd
    from .matching import Matcher
    if not games or not an_odds:
        return []
    gdf = pd.DataFrame(games)
    gdf = gdf[gdf["home_rank"].notna() | gdf["away_rank"].notna()]
    if gdf.empty:
        return []
    m = Matcher(gdf)
    o = pd.DataFrame(an_odds)
    o = o[o["market"] == "spreads"]
    sp = pd.DataFrame(an_splits) if an_splits else pd.DataFrame(columns=["event_id", "market", "side"])
    sp = sp[sp["market"] == "spreads"] if len(sp) else sp
    rows = []
    for eid, d in o.groupby("event_id"):
        r = d.iloc[0]
        gid, sides = m.match(r.home_team, r.away_team, parse_time(r.commence_time), 12)
        if gid is None:
            continue
        g = gdf[gdf.espn_id == gid].iloc[0]
        flip = sides[r.home_team] != "home"          # AN home is ESPN away
        def line(book):
            x = d[(d.book == book) & (d.outcome == r.home_team)]
            return float(x.point.iloc[-1]) if len(x) and x.point.notna().any() else None
        opener, dk = line("an_open"), line("draftkings")
        if opener is None or dk is None:
            continue
        move = dk - opener                            # > 0: AN-home line got worse
        toward = None if abs(move) < 1 else ("away" if move > 0 else "home")
        rec = {"snapshot_ts": iso(now), "rule": RULE, "espn_id": gid, "event_id": eid,
               "commence_time": r.commence_time, "an_home": r.home_team, "an_away": r.away_team,
               "home_rank": g.get("home_rank"), "away_rank": g.get("away_rank"),
               "home_open": opener, "home_dk": dk, "move": move, "toward": toward, "fires": False}
        if toward:
            team = r.home_team if toward == "home" else r.away_team
            odds = d[(d.book == "draftkings") & (d.outcome == team)].price
            s = sp[(sp.event_id == eid) & (sp.side == toward)] if len(sp) else sp
            tk = float(s.tickets_pct.iloc[-1]) if len(s) else None
            mn = float(s.money_pct.iloc[-1]) if len(s) else None
            espn_side = toward if not flip else ("away" if toward == "home" else "home")
            rec.update({"pick": team, "pick_side": espn_side, "pick_line": dk if toward == "home" else -dk,
                        "pick_odds": float(odds.iloc[-1]) if len(odds) else None,
                        "pick_tickets_pct": tk, "pick_money_pct": mn,
                        "fires": tk is not None and mn is not None and 3 <= mn <= 96 and tk < 50 and mn < 50})
        rows.append(rec)
    return rows


# Locked 2026-10-21 from analysis/handle_bands.py (2025 + 2026 backtests); do not edit, add a new version.
SPLIT_RULES = {
    # spread tickets >= 70% on one team but the money majority on the other: back the money side
    "sharp_split_v1": lambda tk, mn: tk >= 70 and mn < 50,
    # spread tickets >= 90% on one team: fade it
    "fade90_v1": lambda tk, mn: tk >= 90,
}


def split_rules_snapshot(games: list[dict], an_odds: list[dict], an_splits: list[dict], now) -> list[dict]:
    """Every game with Action Network spread splits, logged once per locked split rule.
    The 'heavy' team is the one with more spread tickets; both rules bet the other team
    against the DraftKings spread. Money % outside 3-96 is treated as bogus (no bet), as in
    the backtest."""
    import pandas as pd
    from .matching import Matcher
    if not games or not an_odds or not an_splits:
        return []
    gdf = pd.DataFrame(games)
    m = Matcher(gdf)
    o = pd.DataFrame(an_odds)
    o = o[(o["market"] == "spreads") & (o["book"] == "draftkings")]
    sp = pd.DataFrame(an_splits)
    sp = sp[sp["market"] == "spreads"]
    rows = []
    for eid, d in sp.groupby("event_id"):
        r = d.iloc[0]
        gid, sides = m.match(r.home_team, r.away_team, parse_time(r.commence_time), 12)
        if gid is None:
            continue
        flip = sides[r.home_team] != "home"
        by = {x.side: x for x in d.itertuples()}
        if "home" not in by or "away" not in by:
            continue
        tk = {k: to_float(v.tickets_pct) for k, v in by.items()}
        mn = {k: to_float(v.money_pct) for k, v in by.items()}
        if None in tk.values() or None in mn.values():
            continue
        heavy = "home" if tk["home"] >= tk["away"] else "away"
        other = "away" if heavy == "home" else "home"
        team = r.home_team if other == "home" else r.away_team
        line = o[(o.event_id == eid) & (o.outcome == team)]
        pt = to_float(line.point.iloc[-1]) if len(line) else None
        pr = to_float(line.price.iloc[-1]) if len(line) else None
        valid = 3 <= mn[heavy] <= 96 and pt is not None
        for rule, cond in SPLIT_RULES.items():
            rows.append({"snapshot_ts": iso(now), "rule": rule, "espn_id": gid, "event_id": eid,
                         "commence_time": r.commence_time, "an_home": r.home_team, "an_away": r.away_team,
                         "heavy_tickets_pct": tk[heavy], "heavy_money_pct": mn[heavy],
                         "pick": team, "pick_side": other if not flip else heavy,
                         "pick_line": pt, "pick_odds": pr,
                         "fires": bool(valid and cond(tk[heavy], mn[heavy]))})
    return rows


BOOK_RULE = "kalshi_book_fade_v1"   # locked 2026-10-14; do not edit, add a new version instead


def book_rule_snapshot(games: list[dict], k_markets: list[dict], k_books: list[dict],
                       an_odds: list[dict], now) -> list[dict]:
    """Locked rule: fade the team with more resting Kalshi order-book CONTRACTS (bids backing
    it: yes on its ticker + no on the opponent's). Every game with both books this run is
    logged; the bet is the last snapshot before kickoff, graded at the DraftKings spread.
    Dollars, the Kalshi price and DraftKings moneyline are logged too so the result can be
    compared with simply betting the underdog."""
    import pandas as pd
    from .matching import Matcher
    if not games or not k_markets or not k_books:
        return []
    gdf = pd.DataFrame(games)
    m = Matcher(gdf)
    mk = pd.DataFrame(k_markets)
    mk = mk[mk["series"] == "KXNCAAFGAME"]
    tick: dict[str, tuple[str, str]] = {}
    for _, d in mk.groupby("event_ticker"):
        teams = d.drop_duplicates("ticker")[["ticker", "yes_sub_title"]].dropna()
        if len(teams) != 2:
            continue
        a, b = teams["yes_sub_title"].tolist()
        gid, sides = m.match(a, b, parse_time(d["kickoff_est"].dropna().iloc[-1]) if d["kickoff_est"].notna().any()
                             else None, 12)
        if gid is not None:
            for t, name in zip(teams["ticker"], teams["yes_sub_title"]):
                tick[t] = (gid, sides[name])
    mid = {}
    for r in k_markets:
        bid, ask = r.get("yes_bid"), r.get("yes_ask")
        if r.get("ticker") in tick and bid is not None and ask is not None and ask > 0:
            mid[r["ticker"]] = (bid + ask) / 2
    agg: dict[str, dict] = {}
    for r in k_books:
        t = r["ticker"]
        if t not in tick or r.get("size") is None or r.get("price") is None:
            continue
        gid, team = tick[t]
        backs = team if r["side"] == "yes" else ("away" if team == "home" else "home")
        a = agg.setdefault(gid, {"tickers": set(), "home_contracts": 0.0, "away_contracts": 0.0,
                                 "home_usd": 0.0, "away_usd": 0.0})
        a["tickers"].add(t)
        a[f"{backs}_contracts"] += r["size"]
        a[f"{backs}_usd"] += r["size"] * r["price"]
    # DraftKings spread + moneyline per ESPN game
    dk: dict[str, dict] = {}
    o = pd.DataFrame(an_odds)
    if len(o):
        o = o[o["book"] == "draftkings"]
        for _, d in o.groupby("event_id"):
            r0 = d.iloc[0]
            gid, sides = m.match(r0.home_team, r0.away_team, parse_time(r0.commence_time), 12)
            if gid is None:
                continue
            for _, x in d.iterrows():
                side = sides.get(x.outcome)
                if side in ("home", "away") and x.market in ("spreads", "h2h"):
                    dk.setdefault(gid, {})[(x.market, side)] = (x.point, x.price)
    g_by = gdf.set_index("espn_id")
    rows = []
    for gid, a in agg.items():
        if len(a["tickers"]) != 2 or gid not in g_by.index:
            continue
        g = g_by.loc[gid]
        hc, ac = a["home_contracts"], a["away_contracts"]
        pick = None if hc == ac else ("away" if hc > ac else "home")
        ht = [t for t, (i, s) in tick.items() if i == gid and s == "home"]
        line = dk.get(gid, {})
        rec = {"snapshot_ts": iso(now), "rule": BOOK_RULE, "espn_id": gid, "commence_time": g["kickoff"],
               "home": g["home_name"], "away": g["away_name"], "home_rank": g.get("home_rank"),
               "away_rank": g.get("away_rank"), "home_contracts": round(hc, 2), "away_contracts": round(ac, 2),
               "home_usd": round(a["home_usd"], 2), "away_usd": round(a["away_usd"], 2),
               "kalshi_home_p": mid.get(ht[0]) if ht else None, "pick_side": pick,
               "pick": None if pick is None else g[f"{pick}_name"], "fires": False}
        if pick:
            sp, ml = line.get(("spreads", pick)), line.get(("h2h", pick))
            rec.update({"pick_line": sp[0] if sp else None, "pick_odds": sp[1] if sp else None,
                        "pick_ml": ml[1] if ml else None,
                        "pick_is_dog": None if not sp or sp[0] is None else bool(sp[0] > 0),
                        "fires": sp is not None and sp[0] is not None})
        rows.append(rec)
    return rows


def espn_kickoff_lookup(games: list[dict]):
    """(team_a, team_b, approx_time) -> ESPN kickoff, using the analysis name matcher."""
    import pandas as pd
    from .matching import Matcher
    if not games:
        return None
    df = pd.DataFrame(games)
    m = Matcher(df)
    ko = dict(zip(df["espn_id"], df["kickoff"]))

    def find(a, b, when):
        if when is None:
            return None
        gid, _ = m.match(a, b, when, 30)
        return parse_time(ko[gid]) if gid is not None else None
    return find


AN = "https://api.actionnetwork.com/web"
PIN = "https://guest.api.arcadia.pinnacle.com/0.1"
PIN_WEB_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"  # public key the pinnacle.com web app sends
PROBES = {
    "espn": (espn.HOSTS[0] + espn.SCOREBOARD_PATH, {}),
    "dkn_cfb": (dkn.URL + "?tb_eg=87637&tb_edate=n7days&tb_emt=0", {"User-Agent": dkn.UA}),
    "vsin_dk_cfb": (vsin.URL + "?source=DK&sport=CFB", {"User-Agent": vsin.UA}),
    "vsin_circa_cfb": (vsin.URL + "?source=CIRCA&sport=CFB", {"User-Agent": vsin.UA}),
    "kalshi": (kalshi.BASE + "/markets?series_ticker=KXNCAAFGAME&status=open&limit=5", {}),
    "polymarket_gamma": (polymarket.GAMMA + "/sports", {}),
    "espn_summary": (espn.HOSTS[0] + espn.SUMMARY_PATH + "?event=401858242", {}),
    "k_hist_trades_real": (kalshi.BASE + "/historical/trades?ticker=KXNCAAFGAME-26JAN01ORETTU-TTU&limit=5", {}),
    "k_hist_markets_closets": (kalshi.BASE + "/historical/markets?series_ticker=KXNCAAFGAME&limit=3&min_close_ts=1756000000&max_close_ts=1757000000", {}),
    "an_props": (AN + "/v2/games/289385/props?bookIds=15,68,69,75", {}),
    "an_props_osu_books": (AN + "/v2/games/289489/props?bookIds=15,68,69,71,75,79,123,972", {}),
    "an_props_osu_nobooks": (AN + "/v2/games/289489/props", {}),
    "an_props_osu_state": (AN + "/v2/games/289489/props?bookIds=15,68,69,75&stateCode=NJ", {}),
    "an_props_tue_books": (AN + "/v2/games/289875/props?bookIds=15,68,69,71,75,79,123,972", {}),
    "an_props_tue_nobooks": (AN + "/v2/games/289875/props", {}),
    "an_game_osu": (AN + "/v2/games/289489?bookIds=15,68,69", {}),
    "an_game_osu_v1": (AN + "/v1/games/289489", {}),
    "pin_leagues_nokey": (PIN + "/sports/15/leagues?all=false", {}),
    "pin_leagues_key": (PIN + "/sports/15/leagues?all=false", {"x-api-key": PIN_WEB_KEY}),
    "pin_matchups_880": (PIN + "/leagues/880/matchups", {"x-api-key": PIN_WEB_KEY}),
    "pin_markets_880": (PIN + "/leagues/880/markets/straight", {"x-api-key": PIN_WEB_KEY}),
}


def probe(session, save_dir: Path | None = None) -> int:
    """Connectivity check from wherever this runs (e.g. the Actions runner).
    With save_dir, raw response bodies are saved so parsers can be written
    against real payloads."""
    now = utcnow()
    sat = now + timedelta(days=(5 - now.weekday()) % 7)
    fmt = {"today": now.strftime("%Y%m%d"), "sat": sat.strftime("%Y%m%d")}
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, headers) in PROBES.items():
        url = url.format(**fmt)
        try:
            r = session.get(url, headers=headers, timeout=25)
            print(f"{name:22} {r.status_code} {len(r.content):>9}B  {r.text[:120]!r}")
            if save_dir:
                ext = "json" if "json" in r.headers.get("content-type", "") else "txt"
                (save_dir / f"{name}.{ext}").write_bytes(r.content[:3_000_000])
        except Exception as e:
            print(f"{name:22} ERR  {e}")
    # props pipeline check: which games fall in the window and what comes back
    meta: dict = {}
    rows = scraped.an_props_for(session, now, now, now + timedelta(days=7), meta)
    print("props check (7-day window):", meta, "rows:", len(rows))
    cur = scraped.an_current_week(session)
    if save_dir and cur is not None:
        for w in (cur, cur + 1):
            data = session.get(f"{AN}/v2/scoreboard/ncaaf", params={"bookIds": "15", "division": "FBS",
                                                                     "week": w, "season": now.year}, timeout=25)
            (save_dir / f"an_scoreboard_week{w}.json").write_bytes(data.content[:3_000_000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
