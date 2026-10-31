"""DraftKings Network betting splits (free, keyless): DraftKings' own % handle (money) and
% bets (tickets) per selection for moneyline, spread and total, all jurisdictions combined.

  https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=87637&tb_edate=n7days&tb_page=N

tb_eg=87637 is DraftKings' NCAA Football event group. The page is server-rendered and
paginated (about 15 games a page); "n7days" lists every game in the next 7 days, so each
run snapshots the whole week, like the Action Network splits.
"""
from __future__ import annotations

import re
from datetime import datetime
from html import unescape

from ..util import iso, log, to_float

URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
NCAAF = "87637"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
MAX_PAGES = 20


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def _num(s: str) -> float | None:
    s = (s or "").replace("−", "-").replace("–", "-")
    return to_float(s.strip()) if re.fullmatch(r"\s*[+-]?\d+(\.\d+)?\s*", s or "") else None


def parse(page: str, snap: str) -> list[dict]:
    rows = []
    for block in re.split(r'<div class="tb-se[ "]', page)[1:]:
        title = re.search(r'<h5[^>]*>\s*<a[^>]*href="[^"]*/event/(\d+)[^"]*"[^>]*>(.*?)</a>', block, re.S)
        if not title:
            continue
        event_id, matchup = title.group(1), _text(title.group(2))
        if " @ " not in matchup:
            continue
        away, home = (x.strip() for x in matchup.split(" @ ", 1))
        when = re.search(r"</h5>\s*<span[^>]*>(.*?)</span>", block, re.S)
        for mk in re.split(r'<div class="tb-se-head', block)[1:]:
            market = _text(re.search(r'<div class="flex-1">(.*?)</div>', mk, re.S).group(1)).lower()
            if market not in ("moneyline", "spread", "total"):
                continue
            for odd in re.split(r'<div class="tb-sodd', mk)[1:]:
                sel = re.search(r'class="tb-slipline[^"]*">(.*?)</div>', odd, re.S)
                price = re.search(r'class="tb-odd-s[^"]*"[^>]*>(.*?)</a>', odd, re.S)
                pcts = re.findall(r'<div class="flex-1">\s*(\d{1,3})%', odd)
                if not sel or len(pcts) < 2:
                    continue
                label = _text(sel.group(1))
                m = re.match(r"^(.*?)\s+([+-−]?\d+(?:\.\d+)?)$", label) if market != "moneyline" else None
                name, line = (m.group(1), _num(m.group(2))) if m else (label, None)
                rows.append({
                    "snapshot_ts": snap, "dk_event_id": event_id, "away_team": away, "home_team": home,
                    "game_time_et": _text(when.group(1)) if when else None, "market": market,
                    "selection": name, "side": "home" if name == home else ("away" if name == away else name.lower()),
                    "line": line, "odds": _num(_text(price.group(1))) if price else None,
                    "handle_pct": float(pcts[0]), "bets_pct": float(pcts[1]),
                })
    return rows


def fetch(session, now: datetime, meta: dict, group: str = NCAAF, window: str = "n7days") -> list[dict]:
    snap = iso(now)
    out, seen = [], set()
    for page in range(1, MAX_PAGES + 1):
        try:
            r = session.get(URL, params={"tb_eg": group, "tb_edate": window, "tb_emt": "0", "tb_page": page},
                            headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.warning("dk network page %d failed: %s", page, e)
            meta["dkn_error"] = str(e)[:200]
            break
        rows = [x for x in parse(r.text, snap) if (x["dk_event_id"], x["market"], x["selection"]) not in seen]
        if not rows:
            break
        seen.update((x["dk_event_id"], x["market"], x["selection"]) for x in rows)
        out.extend(rows)
    meta["dkn_rows"] = len(out)
    meta["dkn_games"] = len({x["dk_event_id"] for x in out})
    return out
