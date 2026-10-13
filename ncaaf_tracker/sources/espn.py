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


def results(games: list[dict]) -> list[dict]:
    """One clean row per finished game. Only games ESPN marks completed:
    postponed/canceled games also show state 'post' with a 0-0 score."""
    out = []
    for g in games:
        if g.get("completed") is not True:
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        side = "home" if hs > as_ else "away" if as_ > hs else "tie"
        out.append({
            "espn_id": g["espn_id"], "season": g.get("season"), "week": g.get("week"),
            "division": g.get("division"), "kickoff": g.get("kickoff"),
            "home_team": g.get("home_name"), "away_team": g.get("away_name"),
            "home_abbr": g.get("home_abbr"), "away_abbr": g.get("away_abbr"),
            "home_score": hs, "away_score": as_,
            "winner_side": side,
            "winner": g.get(f"{side}_name") if side != "tie" else None,
            "loser": g.get("away_name" if side == "home" else "home_name") if side != "tie" else None,
            "home_margin": hs - as_, "total_points": hs + as_,
            "neutral_site": g.get("neutral_site"), "status_detail": g.get("status_detail"),
        })
    return out


def _stat_values(key: str, display) -> list[tuple[str, float | None, str]]:
    """'completions/passingAttempts' + '22/35' -> two stats; '--' -> None."""
    keys = key.split("/")
    vals = str(display).split("/") if len(keys) > 1 and "/" in str(display) else [display]
    if len(vals) != len(keys):
        return [(key, to_float(display), str(display))]
    return [(k, to_float(v), str(display)) for k, v in zip(keys, vals)]


def parse_box_score(summary: dict, espn_id: str) -> tuple[list[dict], list[dict]]:
    """ESPN game summary -> (team stat rows, player stat rows), one row per stat."""
    box = summary.get("boxscore") or {}
    side = {}
    team_rows, player_rows = [], []
    for t in box.get("teams", []) or []:
        team = t.get("team") or {}
        side[team.get("id")] = t.get("homeAway")
        for st in t.get("statistics", []) or []:
            for name, val, disp in _stat_values(st.get("name", ""), st.get("displayValue")):
                team_rows.append({"espn_id": espn_id, "team_id": team.get("id"),
                                  "team": team.get("displayName"), "home_away": t.get("homeAway"),
                                  "stat": name, "label": st.get("label"), "value": val, "display": disp})
    for p in box.get("players", []) or []:
        team = p.get("team") or {}
        for cat in p.get("statistics", []) or []:
            keys = cat.get("keys") or []
            for a in cat.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                for key, disp in zip(keys, a.get("stats") or []):
                    for name, val, raw in _stat_values(key, disp):
                        player_rows.append({
                            "espn_id": espn_id, "team_id": team.get("id"), "team": team.get("displayName"),
                            "home_away": side.get(team.get("id")), "athlete_id": ath.get("id"),
                            "player": ath.get("displayName"), "jersey": ath.get("jersey"),
                            "category": cat.get("name"), "stat": name, "value": val, "display": raw})
    return team_rows, player_rows


def parse_plays(summary: dict, espn_id: str) -> list[dict]:
    """Every play with ESPN's home win probability after it (summary 'winprobability').

    wp_before is the previous play's wp_after (the first play starts from the
    first entry, which already reflects the pregame line)."""
    wp = {w.get("playId"): w.get("homeWinPercentage") for w in summary.get("winprobability") or []}
    rows, prev = [], None
    if summary.get("winprobability"):
        prev = summary["winprobability"][0].get("homeWinPercentage")
    for d in (summary.get("drives") or {}).get("previous", []) or []:
        for pl in d.get("plays", []) or []:
            start, end = pl.get("start") or {}, pl.get("end") or {}
            after = wp.get(pl.get("id"))
            rows.append({
                "espn_id": espn_id, "play_id": pl.get("id"), "seq": to_float(pl.get("sequenceNumber")),
                "period": (pl.get("period") or {}).get("number"), "clock": (pl.get("clock") or {}).get("displayValue"),
                "type": (pl.get("type") or {}).get("text"), "offense_id": (start.get("team") or {}).get("id"),
                "down": start.get("down"), "distance": start.get("distance"),
                "yards_to_endzone": start.get("yardsToEndzone"), "yards": to_float(pl.get("statYardage")),
                "scoring": pl.get("scoringPlay"), "turnover": pl.get("isTurnover"), "penalty": pl.get("isPenalty"),
                "home_score": pl.get("homeScore"), "away_score": pl.get("awayScore"),
                "home_wp_before": prev, "home_wp_after": after, "text": (pl.get("text") or "")[:200],
            })
            if after is not None:
                prev = after
    return rows


def fetch_box_scores(session, games: list[dict], done_box: set, meta: dict, limit: int = 1000,
                     done_plays: set | None = None):
    """Box scores and play-by-play for completed games not saved yet (finals don't change)."""
    done_plays = done_box if done_plays is None else done_plays
    team_rows, player_rows, play_rows, n, failures = [], [], [], 0, 0
    for g in games:
        gid = g["espn_id"]
        need_box, need_plays = gid not in done_box, gid not in done_plays
        if g.get("completed") is not True or not (need_box or need_plays) or n >= limit:
            continue
        if failures >= 6:
            log.error("espn box scores: repeated failures, stopping this run")
            break
        try:
            data = get_json(session, HOSTS[0] + SUMMARY_PATH, {"event": gid}, min_interval=0.2)
            failures = 0
        except Exception as e:
            failures += 1
            log.warning("espn box score %s failed: %s", gid, e)
            continue
        if need_box:
            t, p = parse_box_score(data, gid)
            team_rows += t
            player_rows += p
        if need_plays:
            play_rows += parse_plays(data, gid)
        n += 1
    meta["box_scores"] = n
    return team_rows, player_rows, play_rows


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
