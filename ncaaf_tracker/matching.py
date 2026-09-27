"""Map each source's team names onto ESPN games.

Every source spells teams differently ("Ohio State Buckeyes", "Ohio St.",
"Ohio State"). Matching = kickoff proximity + best name score for both teams.
Anything that can't be matched confidently is reported, and can be forced via
aliases.csv (raw_name,espn_team_id).
"""
from __future__ import annotations

import csv
import difflib
import re
import unicodedata
from datetime import timedelta
from pathlib import Path

import pandas as pd

ABBREV = {
    "st": "state", "univ": "university", "u": "university", "intl": "international",
    "so": "southern", "n": "north", "s": "south", "e": "east", "w": "west",
    "ga": "georgia", "fla": "florida", "tenn": "tennessee", "mich": "michigan",
    "ky": "kentucky", "la": "louisiana", "miss": "mississippi", "ark": "arkansas",
    "ala": "alabama", "cent": "central", "car": "carolina", "wash": "washington",
}
SYNONYMS = {
    "ole miss": "mississippi", "uconn": "connecticut", "umass": "massachusetts",
    "pitt": "pittsburgh", "hawaii": "hawaii", "miami fl": "miami", "miami florida": "miami",
    "miami ohio": "miami oh", "app state": "appalachian state", "ul monroe": "louisiana monroe",
    "ulm": "louisiana monroe", "southern miss": "southern mississippi",
    "sam houston state": "sam houston", "texas a and m": "texas aandm",
    "university at albany": "albany", "ualbany": "albany",
    "tennessee martin": "ut martin", "se louisiana": "southeastern louisiana",
    "north carolina state": "nc state",
}


def norm(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("é", "e")
    s = re.sub(r"[()'.,\-]", " ", s)
    toks = s.split()
    out = []
    for i, t in enumerate(toks):
        if t == "st" and i == 0:
            out.append("saint")
        else:
            out.append(ABBREV.get(t, t))
    s = " ".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("texas a and m", "texas aandm")
    if s in SYNONYMS:
        return SYNONYMS[s]
    # synonyms also apply with a mascot after them: "umass minutemen" -> "massachusetts minutemen"
    for k in sorted(SYNONYMS, key=len, reverse=True):
        if s.startswith(k + " "):
            return SYNONYMS[k] + s[len(k):]
    return s


def load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {norm(r["raw_name"]): str(r["espn_team_id"]) for r in csv.DictReader(f)
                if r.get("raw_name") and r.get("espn_team_id")}


def team_names(g: pd.Series, side: str) -> list[str]:
    names = [g.get(f"{side}_{k}") for k in ("name", "location", "short", "abbr")]
    return [n for n in (norm(x) for x in names) if n]


def name_score(raw: str, candidates: list[str]) -> float:
    r = norm(raw)
    if not r:
        return 0.0
    best = 0.0
    for c in candidates:
        if r == c:
            return 1.0
        # "ohio state buckeyes" vs location "ohio state"; "alabama" vs "alabama crimson tide"
        if r.startswith(c + " ") or c.startswith(r + " "):
            best = max(best, 0.9 - 0.02 * abs(len(r.split()) - len(c.split())))
        best = max(best, 0.8 * difflib.SequenceMatcher(None, r, c).ratio())
    return best


class Matcher:
    def __init__(self, games: pd.DataFrame, aliases: dict[str, str] | None = None):
        self.games = games.copy()
        self.games["kickoff_dt"] = pd.to_datetime(self.games["kickoff"], utc=True)
        self.aliases = aliases or {}
        self._names = {
            gid: (team_names(g, "home"), team_names(g, "away"), str(g.get("home_id")), str(g.get("away_id")))
            for gid, g in self.games.set_index("espn_id").iterrows()
        }

    def _score(self, raw: str, names: list[str], team_id: str) -> float:
        a = self.aliases.get(norm(raw))
        if a is not None:
            return 1.0 if a == team_id else 0.0
        return name_score(raw, names)

    def match(self, team_a: str, team_b: str, when, window_h: float = 6.0,
              min_score: float = 0.72) -> tuple[str | None, dict | None]:
        """Returns (espn_id, {team_a: 'home'|'away', team_b: ...}) or (None, None)."""
        if when is None or pd.isna(when):
            return None, None
        when = pd.Timestamp(when)
        when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
        # ESPN shows kickoff-TBD games at 04:00Z (midnight ET); widen the window for those.
        ko = self.games["kickoff_dt"]
        tbd = (ko.dt.hour == 4) & (ko.dt.minute == 0)
        diff = (ko - when).abs()
        cand = self.games[(diff <= timedelta(hours=window_h)) | (tbd & (diff <= timedelta(hours=30)))]
        best = (0.0, None, None)
        for _, g in cand.iterrows():
            hn, an, hid, aid = self._names[g["espn_id"]]
            straight = (self._score(team_a, hn, hid), self._score(team_b, an, aid))
            swapped = (self._score(team_a, an, aid), self._score(team_b, hn, hid))
            for (sa, sb), sides in ((straight, ("home", "away")), (swapped, ("away", "home"))):
                if min(sa, sb) < min_score:
                    continue
                total = sa + sb
                if total > best[0]:
                    best = (total, g["espn_id"], {team_a: sides[0], team_b: sides[1]})
        return best[1], best[2]

    def side_of(self, espn_id: str, raw: str) -> str | None:
        hn, an, hid, aid = self._names[espn_id]
        h, a = self._score(raw, hn, hid), self._score(raw, an, aid)
        if max(h, a) < 0.6:
            return None
        return "home" if h >= a else "away"
