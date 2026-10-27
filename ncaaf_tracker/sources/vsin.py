"""VSiN betting splits (free, keyless): DraftKings and Circa handle % / bets % per team for
spread, moneyline and total.

  https://data.vsin.com/betting-splits/?source=DK&sport=CFB      (also source=CIRCA)

The page is server-rendered HTML: a table (class sp-table) with one row per team, columns
named in the header row ("Spread", "Handle %", "Bet %", "ML", ..., "Total", ...). It only
lists the current day's games. Column names are read from the header, so a re-ordered
table still parses; when the page has data we can't parse, the raw HTML is kept for debugging.
"""
from __future__ import annotations

import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser

from ..util import iso, log, to_float

URL = "https://data.vsin.com/betting-splits/"
SOURCES = ("DK", "CIRCA")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


class _Tables(HTMLParser):
    """Collects every <table class="...sp-table..."> as a list of rows of cell texts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._depth = 0          # nesting depth inside an sp-table
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag == "table":
            if self._depth or "sp-table" in cls.split():
                if not self._depth:
                    self.tables.append([])
                self._depth += 1
        elif self._depth:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th") and self._row is not None:
                self._cell = []
            elif tag == "br" and self._cell is not None:
                self._cell.append(" ")

    def handle_endtag(self, tag):
        if not self._depth:
            return
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table":
            self._depth -= 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _pct(v: str) -> float | None:
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", v or "")
    return float(m.group(1)) if m else None


def _num(v: str) -> float | None:
    v = (v or "").replace("½", ".5").replace("PK", "0").replace("pk", "0")
    m = re.search(r"[+-]?\d+(?:\.\d+)?", v)
    return to_float(m.group(0)) if m else None


def _columns(header: list[str]) -> list[str]:
    """Header cells -> field names: spread / spread_handle / spread_bets / ml / ... / total_bets."""
    names, market = [], None
    for h in header:
        k = h.lower()
        if "spread" in k:
            market = "spread"; names.append("spread")
        elif k in ("ml", "moneyline", "money line") or k.startswith("ml"):
            market = "ml"; names.append("ml")
        elif "total" in k:
            market = "total"; names.append("total")
        elif "handle" in k and market:
            names.append(f"{market}_handle")
        elif "bet" in k and market:
            names.append(f"{market}_bets")
        else:
            names.append("team" if not names else f"col{len(names)}")
    return names


def parse(page: str, source: str, snap: str) -> list[dict]:
    p = _Tables()
    p.feed(page)
    rows: list[dict] = []
    for table in p.tables:
        header = next((r for r in table if any("handle" in c.lower() for c in r)), None)
        if not header:
            continue
        names = _columns(header)
        team_rows = []
        for r in table:
            if r is header or len(r) < len(names) - 1:
                continue
            rec = dict(zip(names, r))
            team = re.sub(r"^\s*\(?\d{1,2}\)?\s+", "", rec.get("team", "")).strip()   # drop rank prefix
            if not team or _pct(rec.get("spread_handle", "")) is None and _pct(rec.get("ml_handle", "")) is None:
                continue
            team_rows.append((team, rec))
        # rows come in pairs: away team then home team
        for i in range(0, len(team_rows) - 1, 2):
            (away, ra), (home, rh) = team_rows[i], team_rows[i + 1]
            for side, team, opp, rec in (("away", away, home, ra), ("home", home, away, rh)):
                rows.append({
                    "snapshot_ts": snap, "source": source, "side": side, "team": team, "opponent": opp,
                    "spread": _num(rec.get("spread")), "spread_handle_pct": _pct(rec.get("spread_handle")),
                    "spread_bets_pct": _pct(rec.get("spread_bets")),
                    "ml": _num(rec.get("ml")), "ml_handle_pct": _pct(rec.get("ml_handle")),
                    "ml_bets_pct": _pct(rec.get("ml_bets")),
                    "total": _num(rec.get("total")), "total_handle_pct": _pct(rec.get("total_handle")),
                    "total_bets_pct": _pct(rec.get("total_bets")),
                })
    return rows


def fetch(session, now: datetime, meta: dict, sport: str = "CFB") -> tuple[list[dict], dict[str, str]]:
    """Returns (split rows, {source: raw html} for pages that had data we couldn't parse)."""
    snap = iso(now)
    out, unparsed = [], {}
    for src in SOURCES:
        try:
            r = session.get(URL, params={"source": src, "sport": sport, "display": "table"},
                            headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.warning("vsin %s failed: %s", src, e)
            meta[f"vsin_{src}_error"] = str(e)[:200]
            continue
        page = r.text
        rows = parse(page, src, snap)
        empty = "No splits data available" in unescape(page)
        meta[f"vsin_{src}_rows"] = len(rows)
        if not rows and not empty:
            unparsed[src] = page
            meta[f"vsin_{src}_unparsed"] = True
        out.extend(rows)
    return out, unparsed
