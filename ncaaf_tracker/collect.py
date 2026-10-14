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

from .sources import espn, kalshi, oddsapi, polymarket, scraped
from .util import env, iso, log, make_session, parse_time, utcnow, write_table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("auto", "full", "closing", "backfill", "history"), default="auto")
    ap.add_argument("--data-dir", default=env("DATA_DIR", "data"))
    ap.add_argument("--sources", default="espn,an,pinnacle,odds,kalshi,poly",
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
        lines = []   # one file for all seasons: separate writes in the same second share a name
        for season in sorted({since.year if since.month >= 7 else since.year - 1,
                              until.year if until.month >= 7 else until.year - 1}):
            got = scraped.an_final_lines(session, season)
            meta[f"an_final_lines_{season}"] = len(got)
            lines += got
        write_table(data_dir, "an_final_lines", lines, now)
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
            odds, splits = scraped.an_fetch(session, now, meta)
            tables["an_odds"] = [r for r in odds if in_h(r)]
            tables["an_splits"] = [r for r in splits if in_h(r)]
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
    if "pinnacle" in sources:
        try:
            tables["pinnacle_odds"] = [r for r in scraped.pin_fetch(session, now, meta) if in_h(r)]
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
            tables.update(kalshi.collect(session, now, pm_horizon, meta, kickoff_fn))
        except Exception as e:
            log.exception("kalshi failed")
            meta["kalshi_error"] = str(e)[:300]
    if "poly" in sources:
        try:
            tables.update(polymarket.collect(session, now, pm_horizon, meta, kickoff_fn))
        except Exception as e:
            log.exception("polymarket failed")
            meta["poly_error"] = str(e)[:300]

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
    "kalshi": (kalshi.BASE + "/markets?series_ticker=KXNCAAFGAME&status=open&limit=5", {}),
    "polymarket_gamma": (polymarket.GAMMA + "/sports", {}),
    "espn_summary": (espn.HOSTS[0] + espn.SUMMARY_PATH + "?event=401858242", {}),
    "an_2025_week1": (AN + "/v2/scoreboard/ncaaf?bookIds=15,30,68,69&division=FBS&week=1&season=2025", {}),
    "an_2025_week8": (AN + "/v2/scoreboard/ncaaf?bookIds=15,30,68,69&division=FBS&week=8&season=2025", {}),
    "espn_core_odds_2025": ("https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/401752677/competitions/401752677/odds", {}),
    "espn_summary_2025": (espn.HOSTS[0] + espn.SUMMARY_PATH + "?event=401752677", {}),
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
