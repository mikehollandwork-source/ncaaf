"""Grade the locked rule (ranked_rlm_v1) from the snapshots the collector logs.

  python analysis/track_rule.py --data-dir store/data

A bet is the game's LAST snapshot before kickoff where the rule fired, at the
DraftKings spread and odds logged then. Picks are only counted from the date the
rule was locked (2026-10-14), so nothing here was seen when the rule was chosen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import bands  # noqa: E402
import explosive  # noqa: E402

LOCKED = pd.Timestamp("2026-10-14T00:00:00Z")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    args = ap.parse_args(argv)
    D = Path(args.data_dir)
    p = explosive.load(D, "rule_picks")
    if p.empty:
        print("no rule snapshots yet")
        return 0
    p["snap"] = pd.to_datetime(p.snapshot_ts, utc=True)
    p["ko"] = pd.to_datetime(p.commence_time, utc=True)
    p = p[(p.snap < p.ko) & (p.ko >= LOCKED)].sort_values("snap")
    last = p.groupby("espn_id").tail(1)
    bets = last[last.fires.astype(str) == "True"].copy()
    r = explosive.load(D, "results").drop_duplicates("espn_id", keep="last").set_index("espn_id")
    for c in ("pick_line", "pick_odds"):
        bets[c] = pd.to_numeric(bets[c], errors="coerce")
    rows = []
    for _, b in bets.iterrows():
        if b.espn_id not in r.index:
            rows.append({**b, "result": "pending"})
            continue
        g = r.loc[b.espn_id]
        hs, as_ = float(g.home_score), float(g.away_score)
        margin = (hs - as_) if b.pick_side == "home" else (as_ - hs)
        res = margin + b.pick_line
        odds = -110 if np.isnan(b.pick_odds) else b.pick_odds
        rows.append({**b, "result": "push" if res == 0 else ("win" if res > 0 else "loss"),
                     "profit": 0.0 if res == 0 else (bands.american_profit(odds) if res > 0 else -1.0)})
    out = pd.DataFrame(rows)
    cols = ["commence_time", "an_away", "an_home", "pick", "pick_line", "pick_odds", "pick_tickets_pct",
            "pick_money_pct", "move", "result", "profit"]
    print(f"rule ranked_rlm_v1 since {LOCKED:%Y-%m-%d}: {len(out)} bets")
    if len(out):
        print(out[[c for c in cols if c in out]].to_string(index=False))
        done = out[out.result != "pending"]
        if len(done):
            w, l_ = (done.result == "win").sum(), (done.result == "loss").sum()
            print(f"\nrecord {w}-{l_}-{(done.result == 'push').sum()}  ROI {100 * done.profit.mean():+.1f}%  "
                  f"(break-even at -110 is 52.4%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
