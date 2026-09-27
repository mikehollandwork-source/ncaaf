"""ESPN public scoreboard: schedule, kickoff times, final scores (W/L) and
ESPN's own open/close line (from its odds provider, usually ESPN BET / DraftKings).

Free, no key. Used as the master game list that every other source is matched to.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..util import get_json, iso, log, parse_time, to_float

SCOREBOARD_PATH = "/apis/site/v2/sports/football/college-football/scoreboard"
SUMMARY_PATH = "/apis/site/v2/sports/football/college-football/summary"
# site.api.espn.com 403s GitHub Actions runners; site.web.api works from there.
HOSTS = ["https://site.web.api.espn.com", "https://site.api.espn.com"]
SCOREBOARD = HOSTS[0] + SCOREBOARD_PATH
GROUPS = {"80": "FBS", "81": "FCS"}


def fetch_games(session, start: datetime, end: datetime, groups=("80", "81")) -> list[dict]:
    """One row per game for every day in [start, end]."""
    rows: dict[str, dict] = {}
    day = start.date()
    failures = 0
    while day <= end.date():
        for g in groups:
            if failures >= 3:  # endpoint is down; don't burn the run's time budget
                log.error("espn: 3 consecutive failures, giving up this run")
                return list(rows.values())
            params = {"dates": day.strftime("%Y%m%d"), "groups": g, "limit": 300}  # >300 silently caps at 25
            try:
                data = _scoreboard(session, params)
                failures = 0
            except Exception as e:  # keep going; one bad day shouldn't kill the run
                failures += 1
                log.warning("espn %s group %s failed: %s", day, g, e)
                continue
            for ev in data.get("events", []) or []:
                row = parse_event(ev, GROUPS.get(g, g))
                if row:
                    rows[row["espn_id"]] = row
        day += timedelta(days=1)
    return list(rows.values())


def fill_closed_odds(session, games: list[dict]) -> int:
    """The scoreboard drops odds once a game is final; the per-game summary
    ('pickcenter') still has the provider's open/close. Used by backfill."""
    n = 0
    for g in games:
        if g.get("status_state") != "post" or g.get("espn_details"):
            continue
        try:
            data = get_json(session, HOSTS[0] + SUMMARY_PATH, {"event": g["espn_id"]})
        except Exception as e:
            log.warning("espn summary %s failed: %s", g["espn_id"], e)
            continue
        odds = parse_odds(data.get("pickcenter") or [])
        if odds:
            g.update(odds)
            n += 1
    return n


def _scoreboard(session, params: dict) -> dict:
    """Try each ESPN host; move whichever works to the front for later calls."""
    last = None
    for host in list(HOSTS):
        try:
            data = get_json(session, host + SCOREBOARD_PATH, params)
            if host != HOSTS[0]:
                HOSTS.remove(host)
                HOSTS.insert(0, host)
            return data
        except Exception as e:
            last = e
    raise last


def parse_event(ev: dict, division: str | None = None) -> dict | None:
    comps = ev.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    teams = {c.get("homeAway"): c for c in comp.get("competitors", []) or []}
    home, away = teams.get("home"), teams.get("away")
    if not home or not away:
        return None
    status = (comp.get("status") or ev.get("status") or {}).get("type", {}) or {}
    season = ev.get("season") or {}
    week = ev.get("week") or {}
    row: dict[str, Any] = {
        "espn_id": str(ev.get("id")),
        "season": season.get("year"),
        "season_type": season.get("type"),
        "week": week.get("number"),
        "division": division,
        "kickoff": iso(parse_time(comp.get("date") or ev.get("date"))),
        "name": ev.get("name"),
        "neutral_site": comp.get("neutralSite"),
        "conference_game": comp.get("conferenceCompetition"),
        "status_state": status.get("state"),          # pre / in / post
        "status_detail": status.get("detail"),
        "completed": status.get("completed"),
    }
    for side, c in (("home", home), ("away", away)):
        t = c.get("team") or {}
        row[f"{side}_id"] = t.get("id")
        row[f"{side}_name"] = t.get("displayName")
        row[f"{side}_short"] = t.get("shortDisplayName")
        row[f"{side}_location"] = t.get("location")
        row[f"{side}_nickname"] = t.get("name")
        row[f"{side}_abbr"] = t.get("abbreviation")
        row[f"{side}_score"] = to_float(c.get("score"))
        row[f"{side}_winner"] = c.get("winner")
    row.update(parse_odds(comp.get("odds") or []))
    return row


def _pick(d: Any, *path) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def parse_odds(odds_list: list) -> dict:
    """ESPN odds come in two shapes (older flat and newer open/close blocks).

    Everything is extracted defensively and the raw object is kept so any
    field missed here can be recovered later without re-fetching.
    """
    if not odds_list:
        return {}
    o = odds_list[0]
    out = {
        "espn_odds_provider": _pick(o, "provider", "name"),
        "espn_details": o.get("details"),
        "espn_spread_current": to_float(o.get("spread")),
        "espn_total_current": to_float(o.get("overUnder")),
        "espn_ml_home_current": to_float(_pick(o, "homeTeamOdds", "moneyLine")),
        "espn_ml_away_current": to_float(_pick(o, "awayTeamOdds", "moneyLine")),
    }
    for stage in ("open", "close"):
        out[f"espn_spread_home_{stage}"] = to_float(
            _pick(o, "pointSpread", "home", stage, "line")
            or _pick(o, "homeTeamOdds", stage, "pointSpread", "american")
        )
        out[f"espn_total_{stage}"] = to_float(
            _pick(o, "total", "over", stage, "line")
            or _pick(o, stage, "total", "american")
            or _pick(o, stage, "overUnder")
        )
        out[f"espn_ml_home_{stage}"] = to_float(
            _pick(o, "moneyline", "home", stage, "odds")
            or _pick(o, "homeTeamOdds", stage, "moneyLine", "american")
        )
        out[f"espn_ml_away_{stage}"] = to_float(
            _pick(o, "moneyline", "away", stage, "odds")
            or _pick(o, "awayTeamOdds", stage, "moneyLine", "american")
        )
    out["espn_odds_raw"] = o
    return out
