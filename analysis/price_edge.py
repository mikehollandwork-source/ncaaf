"""Price edges: soft-book prices that beat Pinnacle's no-vig fair price at the same snapshot.

  python analysis/price_edge.py --data-dir store/data [--out reports/price_edge.md]
  python analysis/price_edge.py --data-dir store/data --live [--min-ev 3]   # today's edges

This is the one edge that survived testing: it does not pick winners, it buys the same bet
cheaper than the sharpest book sells it. For spreads, totals and moneylines:

  fair   Pinnacle's two-way price with the vig removed, at Pinnacle's own line
  shift  each extra point vs Pinnacle's line is worth PT_VALUE of win chance
         (spreads 3.0% a point, totals 2.5%); moneylines need no shift
  EV     fair win chance x the soft book's payout, minus the chance of losing

A bet is the FIRST snapshot where a game/market/side shows EV >= the threshold, at the best
book then. Graded on the result, with closing line value against Pinnacle's last line and
bootstrap intervals resampled by game.
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

SOFT = ["draftkings", "fanduel", "betmgm", "caesars", "bet365", "betrivers"]
PT_VALUE = {"spreads": 0.030, "totals": 0.025, "h2h": 0.0}
OUT: list[str] = []
RNG = np.random.default_rng(5)


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def imp(x) -> np.ndarray:
    return np.array([bands.implied(v) if not pd.isna(v) else np.nan for v in x])


def dec(x) -> np.ndarray:
    return np.array([1 + bands.american_profit(v) if not pd.isna(v) else np.nan for v in x])


def snaps(D: Path, m: Matcher, table: str, books: list[str], since: str | None = None) -> pd.DataFrame:
    """Rows per snapshot, ESPN game, market and side (home/away, or over/under)."""
    o = explosive.load(D, table)
    o = o[o.market.isin(PT_VALUE) & o.book.isin(books)]
    if since:
        o = o[o.snapshot_ts >= since]
    o = o.copy()
    for c in ("price", "point"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = P.resolve(o, m, "commence_time")
    o["side"] = np.where(o.market == "totals", o.outcome.str.lower(), P.espn_side(o))
    o["t"] = pd.to_datetime(o.snapshot_ts, utc=True)
    return o[["snapshot_ts", "t", "espn_id", "market", "book", "side", "point", "price"]].dropna(subset=["price"])


def edges(a: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Soft rows joined to Pinnacle's fair price at the same snapshot, with EV per row."""
    key = ["snapshot_ts", "espn_id", "market"]
    p = p.drop_duplicates(key + ["side"], keep="last")
    flip = {"home": "away", "away": "home", "over": "under", "under": "over"}
    o = p.assign(side=p.side.map(flip))[key + ["side", "price"]].rename(columns={"price": "price_other"})
    p = p.merge(o, on=key + ["side"])
    p["p_fair"] = imp(p.price) / (imp(p.price) + imp(p.price_other))
    p = p.rename(columns={"point": "pin_point", "price": "pin_price"})[key + ["side", "pin_point", "pin_price", "p_fair"]]
    j = a.merge(p, on=key + ["side"])
    # extra points in the bettor's favor: a bigger number helps a spread side and an under, hurts an over
    d = (j.point - j.pin_point).fillna(0.0)
    j["d_pts"] = np.where(j.side == "over", -d, d)
    j["p"] = (j.p_fair + j.market.map(PT_VALUE) * j.d_pts).clip(0.02, 0.98)
    j["ev"] = j.p * (dec(j.price) - 1) - (1 - j.p)
    return j


def results(D: Path) -> pd.DataFrame:
    r = explosive.load(D, "results").drop_duplicates("espn_id", keep="last").set_index("espn_id")
    return r[["home_score", "away_score"]].astype(float)


def grade(f: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    f = f[f.espn_id.isin(R.index)].copy()
    hs, as_ = f.espn_id.map(R.home_score), f.espn_id.map(R.away_score)
    margin = np.where(f.side == "home", hs - as_, as_ - hs)
    res = np.select([f.market == "spreads", f.market == "h2h", f.side == "over"],
                    [margin + f.point, margin, hs + as_ - f.point], f.point - (hs + as_))
    f["res"] = res
    f["profit"] = np.where(res == 0, 0.0, np.where(res > 0, dec(f.price) - 1, -1.0))
    return f


def line(d: pd.DataFrame) -> str:
    if d.empty:
        return "n=0"
    w, l_ = (d.res > 0).sum(), (d.res < 0).sum()
    # bootstrap ROI by game: bets on the same game move together
    games = d.groupby("espn_id").profit.agg(["sum", "size"])
    idx = RNG.integers(0, len(games), (2000, len(games)))
    roi = games["sum"].values[idx].sum(1) / games["size"].values[idx].sum(1)
    clv = d.clv.dropna()
    return (f"{w}-{l_} ({100 * w / max(w + l_, 1):.1f}%) ROI {100 * d.profit.mean():+.1f}% "
            f"(95% {100 * np.percentile(roi, 2.5):+.1f} to {100 * np.percentile(roi, 97.5):+.1f}) "
            f"avg EV {100 * d.ev.mean():.1f}% CLV {clv.mean():+.2f} n={len(d)}")


def first_bets(j: pd.DataFrame, thr: float) -> pd.DataFrame:
    f = j[j.ev >= thr].sort_values(["t", "ev"], ascending=[True, False])
    return f.groupby(["espn_id", "market", "side"]).head(1)


def backtest(D: Path) -> None:
    g = P.espn_games(D)
    m = Matcher(g)
    ko = g.set_index("espn_id").kickoff_dt
    div = g.set_index("espn_id").division
    a = snaps(D, m, "an_odds", SOFT)
    p = snaps(D, m, "pinnacle_odds", ["pinnacle"])
    j = edges(a, p)
    j["ko"] = j.espn_id.map(ko)
    j = j[j.t < j.ko]
    j["hrs"] = (j.ko - j.t).dt.total_seconds() / 3600
    R = results(D)
    # closing line value: points (or cents of win chance for moneylines) better than Pinnacle's last line
    last = p.assign(ko=p.espn_id.map(ko))
    last = last[last.t < last.ko].sort_values("t").groupby(["espn_id", "market", "side"]).tail(1)
    lp = last.set_index(["espn_id", "market", "side"])
    cl_pt, cl_pr = lp.point, lp.price
    say(f"# Price edges vs Pinnacle ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d})")
    say(f"- snapshots with both a soft book and Pinnacle: {j.snapshot_ts.nunique()} | games {j.espn_id.nunique()} | "
        f"from {j.snapshot_ts.min()[:10]} to {j.snapshot_ts.max()[:10]}")

    def prep(f):
        f = grade(f, R)
        k = list(zip(f.espn_id, f.market, f.side))
        cp = np.array([cl_pt.get(x, np.nan) for x in k], float)
        d = f.point.values - cp
        pts = np.where(f.side == "over", -d, d)
        mlc = 100 * (imp([cl_pr.get(x, np.nan) for x in k]) - imp(f.price))    # moneyline: implied-chance gain
        f["clv"] = np.where(f.market == "h2h", mlc, pts)
        f["div"] = f.espn_id.map(div)
        return f

    for mk, name in (("spreads", "Spreads"), ("totals", "Totals"), ("h2h", "Moneylines")):
        jm = j[j.market == mk]
        say(f"\n## {name}")
        base = prep(jm.sort_values("t").groupby(["espn_id", "book", "side"]).head(1))
        say(f"- guard, every book's first line, no filter: {line(base)}")
        for thr in (0.01, 0.02, 0.03, 0.05):
            say(f"- EV >= {100 * thr:.0f}%: {line(prep(first_bets(jm, thr)))}")
        f = prep(first_bets(jm, 0.02))
        if f.empty:
            continue
        say("- EV >= 2%, split:")
        for lab, msk in (("48h+ before kickoff", f.hrs >= 48), ("12-48h", f.hrs.between(12, 48, inclusive="left")),
                         ("under 12h", f.hrs < 12), ("FBS", f["div"] == "FBS"), ("FCS", f["div"] == "FCS")):
            say(f"  - {lab}: {line(f[msk])}")
        if mk != "h2h":
            for lab, msk in (("same line, better price", f.d_pts == 0), ("half point better", f.d_pts.between(0.4, 0.6)),
                             ("1+ points better", f.d_pts >= 0.9)):
                say(f"  - {lab}: {line(f[msk])}")
        else:
            for lab, msk in (("favorites", f.price < 0), ("dogs to +250", f.price.between(100, 250)), ("dogs +251 and up", f.price > 250)):
                say(f"  - {lab}: {line(f[msk])}")
        for b, d in f.groupby("book"):
            say(f"  - {b}: {line(d)}")


def live(D: Path, min_ev: float) -> None:
    now = pd.Timestamp.now(tz="UTC")
    g = P.espn_games(D)
    g = g[(g.kickoff_dt > now) & (g.kickoff_dt < now + pd.Timedelta(days=7))]
    m = Matcher(g)
    since = (now - pd.Timedelta(hours=6)).isoformat()
    a, p = snaps(D, m, "an_odds", SOFT, since), snaps(D, m, "pinnacle_odds", ["pinnacle"], since)
    if a.empty or p.empty:
        print("no snapshot in the last 6 hours")
        return
    ts = min(a.snapshot_ts.max(), p.snapshot_ts.max())
    j = edges(a[a.snapshot_ts == ts], p[p.snapshot_ts == ts])
    j = j[j.ev >= min_ev / 100].sort_values("ev", ascending=False).groupby(["espn_id", "market", "side"]).head(1)
    nm = g.set_index("espn_id")
    print(f"price edges at {ts} (EV >= {min_ev:g}% vs Pinnacle no-vig): {len(j)}")
    for r in j.sort_values("ev", ascending=False).itertuples():
        e = nm.loc[r.espn_id]
        who = {"home": e.home_short, "away": e.away_short}.get(r.side, r.side)
        pt = "" if r.market == "h2h" else f" {r.point:+g}" if r.market == "spreads" else f" {r.point:g}"
        pin = "" if r.market == "h2h" else f" {r.pin_point:+g}" if r.market == "spreads" else f" {r.pin_point:g}"
        print(f"  EV {100 * r.ev:4.1f}%  {e.kickoff_dt.tz_convert('America/New_York'):%a %-I:%M %p}  {e.away_short} @ {e.home_short}: "
              f"{who}{pt} ({r.price:+.0f}) at {r.book} | Pinnacle{pin} ({r.pin_price:+.0f}), fair {100 * r.p_fair:.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--min-ev", type=float, default=3.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    D = Path(a.data_dir)
    if a.live:
        live(D, a.min_ev)
        return 0
    backtest(D)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(OUT) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
