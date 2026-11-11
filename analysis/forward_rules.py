"""Two rules locked 2026-11-06 from the seven-way pattern search, tracked forward only.

  luck_fade_v1   Back the team that has been 7+ pts/game LESS lucky than its opponent this season
                 (luck = actual margin minus the margin its play-by-play success rate predicts,
                 129.5 pts per 1.0 of success-rate differential, frozen). Both teams need 3+
                 games with play-by-play. Held in both seasons out of sample (54.5% / 57.3%).
  luck_fade_fcs_v1  The same rule in FCS games only (locked 2026-11-07, after the noon kickoffs).
                 The luck edge lives in FCS: 7+ gap 56.5% (2025) / 64.4% (2026), while FBS games
                 were 52.1% / 51.8%. Tracked alongside v1; v1 is not edited.
  late_surge_fade_v1  INFO ONLY (locked 2026-11-08 03:00Z). One side's share of the spread money
                 jumped 20+ pts to 60%+ in the final 24h (last valid split 24h+ out vs the last
                 one before kickoff) but the DraftKings line did not move toward it: bet the other
                 side. 2026 backtest 10-6 on only 16 games; random samples matched it 36% of the
                 time, so it is tracked, not bet.
  tight_fbs_fade_v1, tight_fcs_back_v1  INFO ONLY (locked 2026-11-11 17:00Z). Games with a
                 DraftKings spread under 7 where one team has 70%+ of the spread money at the last
                 snapshot before kickoff. FBS: bet the OTHER team (56.1% in 2025, 63.3% in 2026).
                 FCS: bet the heavy team (57.7% / 59.4%). One of ~20 cuts and only ~30 games
                 a side in 2026, so both are tracked, not bet.
  early_rlm_v1   DraftKings spread moved 1+ pt toward a team, most of that move came earlier than
                 24h before kickoff, and under 50% of the spread money is on that team.
                 2026 only (63.5%, n=53), so it is the weaker of the two.

Every input is known before kickoff (only completed games feed luck; only pre-kickoff
snapshots feed the line and money), so grading is deterministic and look-ahead free. The bet
is the last pre-kickoff DraftKings spread and price. Each pick also shows the best line on
the board across the soft books next to Pinnacle's number (line shopping held up as real).

  python analysis/forward_rules.py --data-dir store/data [--hours 48]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import bands  # noqa: E402
import explosive  # noqa: E402
import patterns as P  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402

LOCK = pd.Timestamp("2026-11-06T04:00:00Z")
FCS_LOCK = pd.Timestamp("2026-11-07T17:30:00Z")    # after the Nov 7 noon kickoffs it was chosen from
SURGE_LOCK = pd.Timestamp("2026-11-08T03:00:00Z")
RULE_LOCK = {"luck_fade_v1": LOCK, "early_rlm_v1": LOCK, "luck_fade_fcs_v1": FCS_LOCK,
             "late_surge_fade_v1": SURGE_LOCK}
TIGHT_LOCK = pd.Timestamp("2026-11-11T17:00:00Z")
RULE_LOCK.update({"tight_fbs_fade_v1": TIGHT_LOCK, "tight_fcs_back_v1": TIGHT_LOCK})
INFO_ONLY = {"late_surge_fade_v1", "tight_fbs_fade_v1", "tight_fcs_back_v1"}
RULES = tuple(RULE_LOCK)
LUCK_SLOPE = 129.5          # frozen; fit on 2025+2026 to Nov 6 (127.4 and 132.3 by season)
LUCK_GAP = 7.0
LUCK_MIN_GAMES = 3
SOFT = ["draftkings", "fanduel", "betmgm", "caesars", "bet365", "betrivers"]


def games(D: Path) -> pd.DataFrame:
    g = P.espn_games(D)
    g["season"] = pd.to_numeric(g.season, errors="coerce")
    return g


def spreads(D: Path, m: Matcher, table: str, books: list[str], since: pd.Timestamp) -> pd.DataFrame:
    """Spread rows per ESPN game and ESPN side (home/away): book, snapshot time, point, price."""
    o = explosive.load(D, table)
    o = o[(o.market == "spreads") & o.book.isin(books) & (o.snapshot_ts >= since.isoformat())].copy()
    if o.empty:
        return o
    for c in ("point", "price"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = P.resolve(o, m, "commence_time")
    o["eside"] = P.espn_side(o)
    o["t"] = pd.to_datetime(o.snapshot_ts, utc=True)
    return o[["espn_id", "book", "eside", "t", "point", "price"]].dropna(subset=["point"])


def money(D: Path, m: Matcher, since: pd.Timestamp) -> pd.DataFrame:
    sp = explosive.load(D, "an_splits")
    sp = sp[(sp.market == "spreads") & (sp.snapshot_ts >= since.isoformat())].copy()
    sp = P.resolve(sp, m, "commence_time")
    sp["eside"] = np.where((sp.side == "home") != sp.flipped, "home", "away")
    sp["money_pct"] = pd.to_numeric(sp.money_pct, errors="coerce")
    sp["t"] = pd.to_datetime(sp.snapshot_ts, utc=True)
    return sp[["espn_id", "eside", "t", "money_pct"]]


# ------------------------------------------------------------------ luck_fade_v1

def luck_rule(D: Path, G: pd.DataFrame) -> pd.DataFrame:
    t = P.team_stats(D, slope=LUCK_SLOPE)[["team", "season", "kickoff", "luck"]]
    by = {k: d.sort_values("kickoff") for k, d in t.groupby(["team", "season"])}

    def pre(team, season, ko):
        d = by.get((str(team), season))
        if d is None:
            return np.nan, 0
        d = d[d.kickoff < ko]
        return (d.luck.mean() if len(d) else np.nan), len(d)

    rows = []
    for _, g in G.iterrows():
        (lh, nh), (la, na) = pre(g.home_id, g.season, g.kickoff_dt), pre(g.away_id, g.season, g.kickoff_dt)
        if min(nh, na) < LUCK_MIN_GAMES:
            continue
        gap = lh - la                                     # home luck minus away luck
        side = "away" if gap >= LUCK_GAP else ("home" if gap <= -LUCK_GAP else None)
        rows.append({"espn_id": g.espn_id, "rule": "luck_fade_v1", "pick_side": side,
                     "fires": side is not None, "note": f"luck gap {abs(gap):.1f} pts/g"})
        if g.division == "FCS":
            rows.append({**rows[-1], "rule": "luck_fade_fcs_v1"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ early_rlm_v1

def tight_heavy_rule(G: pd.DataFrame, L: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    g = G.set_index("espn_id")
    M = M[M.money_pct.between(3, 96)]
    dk = L[L.book == "draftkings"]
    rows = []
    for (gid, side), mm in M.groupby(["espn_id", "eside"]):
        if gid not in g.index:
            continue
        k0, div = g.kickoff_dt[gid], g.division[gid]
        mm = mm[mm.t < k0].sort_values("t")
        ln = dk[(dk.espn_id == gid) & (dk.eside == side) & (dk.t < k0)].sort_values("t")
        if mm.empty or ln.empty or div not in ("FBS", "FCS"):
            continue
        mon, line = mm.money_pct.iloc[-1], ln.point.iloc[-1]
        if mon < 70 or abs(line) >= 7:
            continue
        fade = div == "FBS"
        rows.append({"espn_id": gid, "rule": "tight_fbs_fade_v1" if fade else "tight_fcs_back_v1",
                     "pick_side": ("away" if side == "home" else "home") if fade else side, "fires": True,
                     "note": f"{mon:.0f}% of spread money on the {'other' if fade else 'picked'} team, "
                             f"spread {abs(line):g} (INFO ONLY)"})
    return pd.DataFrame(rows, columns=["espn_id", "rule", "pick_side", "fires", "note"])


def late_surge_rule(G: pd.DataFrame, L: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    ko = G.set_index("espn_id").kickoff_dt
    M = M[M.money_pct.between(3, 96)]
    dk = L[L.book == "draftkings"]
    rows = []
    for (gid, side), mm in M.groupby(["espn_id", "eside"]):
        k0 = ko.get(gid)
        if k0 is None:
            continue
        mm = mm[mm.t < k0].sort_values("t")
        early = mm[mm.t <= k0 - pd.Timedelta(hours=24)]
        if early.empty or len(mm) < 2:
            continue
        e, c = early.iloc[-1], mm.iloc[-1]
        ln = dk[(dk.espn_id == gid) & (dk.eside == side) & (dk.t < k0)].sort_values("t")
        le = ln[ln.t <= e.t]
        if ln.empty or le.empty:
            continue
        toward = -(ln.point.iloc[-1] - le.point.iloc[-1])   # > 0: line moved toward the surge side
        surge = c.money_pct - e.money_pct
        fires = surge >= 20 and c.money_pct >= 60 and toward <= 0
        if fires:
            rows.append({"espn_id": gid, "rule": "late_surge_fade_v1", "pick_side": "away" if side == "home" else "home",
                         "fires": True, "note": f"money on the other side {e.money_pct:.0f}% -> {c.money_pct:.0f}%, "
                                                f"line moved {toward + 0.0:+g} toward it (INFO ONLY)"})
    return pd.DataFrame(rows, columns=["espn_id", "rule", "pick_side", "fires", "note"])


def early_rlm_rule(G: pd.DataFrame, L: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    ko = G.set_index("espn_id").kickoff_dt
    dk = L[(L.book == "draftkings") & (L.eside == "home")]
    rows = []
    for gid, d in dk.groupby("espn_id"):
        k0 = ko.get(gid)
        d = d[d.t < k0].sort_values("t")
        if len(d) < 2:
            continue
        first, last = d.point.iloc[0], d.point.iloc[-1]
        before = d[d.t <= k0 - pd.Timedelta(hours=24)]
        at24 = before.point.iloc[-1] if len(before) else first
        total = last - first
        if abs(total) < 1:
            rows.append({"espn_id": gid, "rule": "early_rlm_v1", "pick_side": None, "fires": False,
                         "note": f"moved {total:+g}"})
            continue
        side = "home" if total < 0 else "away"
        late_share = (last - at24) / total
        mm = M[(M.espn_id == gid) & (M.eside == side) & (M.t < k0)].sort_values("t")
        mon = mm.money_pct.iloc[-1] if len(mm) else np.nan
        ok = late_share < 0.5 and 3 <= mon < 50
        # until the final 24h starts, every move counts as early: the call is only final at kickoff
        open_window = d.t.iloc[-1] < k0 - pd.Timedelta(hours=24)
        rows.append({"espn_id": gid, "rule": "early_rlm_v1", "pick_side": side if ok else None, "fires": bool(ok),
                     "note": f"moved {abs(total):g} toward {side}, {100 * (1 - late_share):.0f}% early, "
                             f"money {mon:.0f}%" + (" (provisional: a late move can still cancel it)"
                                                    if open_window else "")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pricing

def price_picks(picks: pd.DataFrame, G: pd.DataFrame, L: pd.DataFrame, PIN: pd.DataFrame) -> pd.DataFrame:
    """DraftKings line at the last pre-kickoff snapshot (the graded bet), plus the best soft-book
    line and Pinnacle's line at that book's latest pre-kickoff snapshot."""
    ko = G.set_index("espn_id").kickoff_dt
    out = []
    for _, p in picks.iterrows():
        k0 = ko[p.espn_id]
        x = L[(L.espn_id == p.espn_id) & (L.eside == p.pick_side) & (L.t < k0)]
        if x.empty:
            continue
        latest = x.sort_values("t").groupby("book").tail(1)
        dk = latest[latest.book == "draftkings"]
        if dk.empty:
            continue
        best = latest.assign(pr=latest.price.fillna(-110)).sort_values(["point", "pr"], ascending=False).iloc[0]
        pn = PIN[(PIN.espn_id == p.espn_id) & (PIN.eside == p.pick_side) & (PIN.t < k0)].sort_values("t")
        out.append({**p, "pick_line": dk.point.iloc[0], "pick_odds": dk.price.iloc[0],
                    "best_book": best.book, "best_line": best.point, "best_odds": best.price,
                    "pin_line": pn.point.iloc[-1] if len(pn) else np.nan})
    return pd.DataFrame(out)


def build(D: Path, since: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    G = games(D)
    G = G[G.kickoff_dt >= since]
    m = Matcher(G)
    lo = since - pd.Timedelta(days=10)
    L = spreads(D, m, "an_odds", SOFT, lo)
    PIN = spreads(D, m, "pinnacle_odds", ["pinnacle"], lo)
    M = money(D, m, lo)
    picks = pd.concat([luck_rule(D, G), early_rlm_rule(G, L, M), late_surge_rule(G, L, M),
                       tight_heavy_rule(G, L, M)], ignore_index=True)
    picks = picks[picks.fires]
    if len(picks):                                    # each rule counts only games after its own lock
        ko = G.set_index("espn_id").kickoff_dt
        picks = picks[[ko[e] >= RULE_LOCK[r] for e, r in zip(picks.espn_id, picks.rule)]]
    picks = price_picks(picks, G, L, PIN) if len(picks) else picks
    if len(picks):
        name = G.set_index("espn_id")
        picks["ko"] = picks.espn_id.map(name.kickoff_dt)
        picks["game"] = picks.espn_id.map(lambda e: f"{name.away_short[e]} @ {name.home_short[e]}")
        picks["pick"] = [name.home_short[e] if s == "home" else name.away_short[e]
                         for e, s in zip(picks.espn_id, picks.pick_side)]
    return picks, G


def grade(picks: pd.DataFrame, D: Path) -> pd.DataFrame:
    r = explosive.load(D, "results").drop_duplicates("espn_id", keep="last").set_index("espn_id")
    b = picks[picks.espn_id.isin(r.index)].copy()
    if b.empty:
        return b
    hs, as_ = b.espn_id.map(r.home_score).astype(float), b.espn_id.map(r.away_score).astype(float)
    b["margin"] = np.where(b.pick_side == "home", hs - as_, as_ - hs)
    res = b.margin + b.pick_line
    b["result"] = np.select([res > 0, res < 0], ["win", "loss"], "push")
    b["profit"] = [0.0 if x == 0 else (bands.american_profit(-110 if pd.isna(o) else o) if x > 0 else -1.0)
                   for x, o in zip(res, b.pick_odds)]
    b["best_res"] = b.margin + b.best_line
    return b


def fmt_line(x, o) -> str:
    return f"{x:+g} ({'' if pd.isna(o) else f'{o:+.0f}'})"


def report(D: Path, hours: float = 48) -> None:
    picks, _ = build(D, LOCK)
    done = grade(picks, D) if len(picks) else picks
    now = pd.Timestamp.now(tz="UTC")
    for rule in RULES:
        b = done[done.rule == rule] if len(done) else done
        tag = " (INFO ONLY, not a bet)" if rule in INFO_ONLY else ""
        print(f"\nrule {rule}{tag} since {RULE_LOCK[rule]:%Y-%m-%d %H:%M}Z: {len(b)} graded bets")
        if len(b):
            w, l_, p_ = (b.result == "win").sum(), (b.result == "loss").sum(), (b.result == "push").sum()
            bw, bl = (b.best_res > 0).sum(), (b.best_res < 0).sum()
            print(f"  ATS {w}-{l_}-{p_} ROI {100 * b.profit.mean():+.1f}% at DraftKings | at the best book {bw}-{bl}")
            print(b[["ko", "game", "pick", "pick_line", "best_book", "best_line", "margin", "result", "note"]]
                  .to_string(index=False))
    up = picks[(picks.ko > now) & (picks.ko <= now + pd.Timedelta(hours=hours))] if len(picks) else picks
    print(f"\nupcoming picks, next {hours:g}h: {len(up)}")
    for _, p in up.sort_values("ko").iterrows():
        pin = "" if pd.isna(p.pin_line) else f" | Pinnacle {p.pin_line:+g}"
        print(f"  {p.ko.tz_convert('America/New_York'):%a %-I:%M %p}  {p.game:32} {p.rule:13} {p.pick} "
              f"DK {fmt_line(p.pick_line, p.pick_odds)} | best {p.best_book} {fmt_line(p.best_line, p.best_odds)}"
              f"{pin} | {p.note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--hours", type=float, default=48)
    a = ap.parse_args(argv)
    report(Path(a.data_dir), a.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
