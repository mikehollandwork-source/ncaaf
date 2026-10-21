"""Lopsided public handles: back or fade the heavy side, at every tickets/money band.

  python analysis/handle_bands.py --data-dir store/data

For each game the "heavy" side is the team with the larger share. Spread splits are graded
against the spread, moneyline splits on the moneyline (and ATS for reference), both seasons,
at closing DraftKings prices (consensus if missing).
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
import backtest as bt  # noqa: E402
import edge_search as es  # noqa: E402

EDGES = [50, 60, 70, 80, 90, 101]
LABELS = ["50-60", "60-70", "70-80", "80-90", "90+"]


def grade(d: pd.DataFrame, home: pd.Series) -> dict:
    margin = np.where(home, d.hs - d["as"], d["as"] - d.hs)
    line = np.where(home, d.sp_close, -d.sp_close)
    res = margin + line
    sp_o = np.where(home, d.sp_h_odds, d.sp_a_odds)
    ml_o = np.where(home, d.ml_h_close, d.ml_a_close)
    sp_p = [0.0 if r == 0 else (bands.american_profit(-110 if np.isnan(o) else o) if r > 0 else -1.0)
            for r, o in zip(res, sp_o)]
    ml_p = [bands.american_profit(o) if m > 0 else -1.0 for m, o in zip(margin, ml_o) if not np.isnan(o)]
    w, l_ = int((res > 0).sum()), int((res < 0).sum())
    return {"n": len(d), "ats": f"{w}-{l_}", "cov%": 100 * w / max(w + l_, 1), "atsROI": 100 * np.mean(sp_p) if sp_p else np.nan,
            "win%": 100 * np.mean(margin > 0), "mlROI": 100 * np.mean(ml_p) if ml_p else np.nan}


def table(g: pd.DataFrame, market: str, metric: str) -> pd.DataFrame:
    hv = g[f"{market}_{metric}_home"]
    av = g[f"{market}_{metric}_away"]
    heavy_home = hv >= av
    share = np.where(heavy_home, hv, av)
    band = pd.cut(pd.Series(share, index=g.index), EDGES, labels=LABELS, right=False)
    rows = []
    for season in [2025, 2026, "both"]:
        s = g if season == "both" else g[g.season == season]
        for lab in LABELS:
            m = (band.loc[s.index] == lab)
            d = s[m]
            if len(d) < 10:
                continue
            for act, home in (("back heavy", heavy_home.loc[d.index]), ("fade heavy", ~heavy_home.loc[d.index])):
                rows.append({"season": season, "band": lab, "bet": act, **grade(d, home.values)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    D = Path(ap.parse_args(argv).data_dir)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    g = bt.games(D)
    g = g[g.sp_close.notna()].copy()
    sp = es.splits(D)
    w = sp.pivot_table(index="espn_id", columns="side", values=["sp_tix", "sp_mon", "ml_tix", "ml_mon"])
    w.columns = [f"{a.split('_')[0]}_{a.split('_')[1]}_{b}" for a, b in w.columns]
    g = g.merge(w.reset_index(), on="espn_id", how="inner")
    print(f"games with splits: {g.season.value_counts().sort_index().to_dict()}")
    fmt = {"cov%": "{:.1f}", "atsROI": "{:+.1f}", "win%": "{:.1f}", "mlROI": "{:+.1f}"}
    for market, name in (("sp", "SPREAD splits"), ("ml", "MONEYLINE splits")):
        for metric in ("tix", "mon"):
            t = table(g.dropna(subset=[f"{market}_{metric}_home", f"{market}_{metric}_away"]), market, metric)
            for c, f in fmt.items():
                t[c] = t[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
            print(f"\n=== {name}, by {'tickets' if metric == 'tix' else 'money'} % on the heavy side ===")
            print(t.to_string(index=False))
    # both tickets AND money lopsided on the same team
    print("\n=== SPREAD: tickets AND money both >= X% on the same team ===")
    rows = []
    for x in (60, 70, 80, 90):
        hh = (g.sp_tix_home >= x) & (g.sp_mon_home >= x)
        aa = (g.sp_tix_away >= x) & (g.sp_mon_away >= x)
        for season in [2025, 2026, "both"]:
            sm = (g.season == season) if season != "both" else pd.Series(True, index=g.index)
            d = g[(hh | aa) & sm]
            if len(d) < 10:
                continue
            heavy = hh[d.index].values
            for act, home in (("back heavy", heavy), ("fade heavy", ~heavy)):
                rows.append({"both>=": x, "season": season, "bet": act, **grade(d, home)})
    t = pd.DataFrame(rows)
    for c, f in fmt.items():
        t[c] = t[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
    print(t.to_string(index=False))
    # sharp split: tickets heavy one way, money heavy the other
    print("\n=== SPREAD: tickets >= X% on one team but money majority on the OTHER (big bets vs public) ===")
    rows = []
    for x in (60, 70, 80):
        th = (g.sp_tix_home >= x) & (g.sp_mon_home < 50)
        ta = (g.sp_tix_away >= x) & (g.sp_mon_away < 50)
        for season in [2025, 2026, "both"]:
            sm = (g.season == season) if season != "both" else pd.Series(True, index=g.index)
            d = g[(th | ta) & sm]
            if len(d) < 10:
                continue
            money_home = ta[d.index].values         # money side = opposite of the ticket side
            for act, home in (("back money side", money_home), ("back ticket side", ~money_home)):
                rows.append({"tix>=": x, "season": season, "bet": act, **grade(d, home)})
    t = pd.DataFrame(rows)
    for c, f in fmt.items():
        t[c] = t[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
