"""One collection pass. Run on a schedule (see .github/workflows/collect.yml).

Modes
  full     every source, every pre-kickoff game (the hourly snapshot)
  closing  only games kicking off within CLOSING_WINDOW_MIN; exits without
           writing if there are none. Tightens the closing-line snapshot.
  auto     full if minute < 15, else closing (default)
  backfill ESPN schedule/results/open-close lines only, from --since to today
           (use once to pick up games played before tracking started)

Tracking stops at kickoff: every source filters to games that haven't started
(prediction markets with a +/-1h lenient window because their kickoff time is
estimated; analysis then trims to the exact ESPN kickoff).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

from .sources import espn, kalshi, oddsapi, polymarket
from .util import env, iso, log, make_session, parse_time, utcnow, write_table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("auto", "full", "closing", "backfill"), default="auto")
    ap.add_argument("--data-dir", default=env("DATA_DIR", "data"))
    ap.add_argument("--sources", default="espn,odds,kalshi,poly",
                    help="comma list of sources to poll")
    ap.add_argument("--since", help="backfill start date YYYY-MM-DD")
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
        return probe(session)

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
    if mode == "backfill":
        since = parse_time(f"{args.since}T00:00:00Z") if args.since else now - timedelta(days=45)
        games = espn.fetch_games(session, since, now + timedelta(days=14))
        for g in games:
            g["snapshot_ts"] = iso(now)
        write_table(data_dir, "espn_games", games, now)
        log.info("backfilled %d games since %s", len(games), iso(since))
        return 0
    if mode == "full":
        games = espn.fetch_games(session, now - timedelta(days=3), now + timedelta(days=14))
        for g in games:
            g["snapshot_ts"] = iso(now)
        tables["espn_games"] = games
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

    # ---- sportsbooks
    if "odds" in sources:
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
            tables.update(kalshi.collect(session, now, pm_horizon, meta))
        except Exception as e:
            log.exception("kalshi failed")
            meta["kalshi_error"] = str(e)[:300]
    if "poly" in sources:
        try:
            tables.update(polymarket.collect(session, now, pm_horizon, meta))
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


PROBES = {
    "espn": espn.SCOREBOARD,
    "espn_alt": espn.HOSTS[1] + espn.SCOREBOARD_PATH,
    "odds_api": "https://api.the-odds-api.com/v4/sports",
    "kalshi": kalshi.BASE + "/markets?series_ticker=KXNCAAFGAME&status=open&limit=5",
    "polymarket_gamma": polymarket.GAMMA + "/sports",
    "polymarket_clob": polymarket.CLOB + "/time",
    "polymarket_data": polymarket.DATA + "/trades?limit=1",
}


def probe(session) -> int:
    """Connectivity check from wherever this runs (e.g. the Actions runner)."""
    for name, url in PROBES.items():
        try:
            r = session.get(url, timeout=20)
            print(f"{name:18} {r.status_code}  {r.text[:160]!r}")
        except Exception as e:
            print(f"{name:18} ERR  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
