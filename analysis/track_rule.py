"""Grade the locked rules from the snapshots the collector logs:
  ranked_rlm_v1        (table rule_picks)
  kalshi_book_fade_v1  (table book_picks; also shows "bet every underdog" on the same games,
                        since fading book money is mostly an underdog bet)

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
    book(D)
    return 0


def grade(D: Path, table: str) -> pd.DataFrame:
    p = explosive.load(D, table)
    if p.empty:
        return p
    p["snap"] = pd.to_datetime(p.snapshot_ts, utc=True)
    p["ko"] = pd.to_datetime(p.commence_time, utc=True)
    p = p[(p.snap < p.ko) & (p.ko >= LOCKED)].sort_values("snap")
    bets = p.groupby("espn_id").tail(1)
    bets = bets[bets.fires.astype(str) == "True"].copy()
    for c in ("pick_line", "pick_odds", "pick_ml"):
        if c in bets:
            bets[c] = pd.to_numeric(bets[c], errors="coerce")
    r = explosive.load(D, "results").drop_duplicates("espn_id", keep="last").set_index("espn_id")
    bets = bets[bets.espn_id.isin(r.index)].copy()
    hs = bets.espn_id.map(r.home_score).astype(float)
    as_ = bets.espn_id.map(r.away_score).astype(float)
    bets["margin"] = np.where(bets.pick_side == "home", hs - as_, as_ - hs)
    return bets


def show(b: pd.DataFrame, label: str) -> None:
    res = b.margin + b.pick_line
    sp = [0.0 if x == 0 else (bands.american_profit(-110 if np.isnan(o) else o) if x > 0 else -1.0)
          for x, o in zip(res, b.pick_odds)]
    ml = [bands.american_profit(o) if m > 0 else -1.0 for m, o in zip(b.margin, b.pick_ml) if not np.isnan(o)]
    print(f"  {label:28} ATS {(res > 0).sum()}-{(res < 0).sum()}-{(res == 0).sum()} ROI "
          f"{100 * np.mean(sp) if sp else float('nan'):+.1f}% | ML {(b.margin > 0).sum()}-{(b.margin < 0).sum()} "
          f"ROI {100 * np.mean(ml) if ml else float('nan'):+.1f}%")


def book(D: Path) -> None:
    b = grade(D, "book_picks")
    print(f"\nrule kalshi_book_fade_v1 since {LOCKED:%Y-%m-%d}: {len(b)} graded bets")
    if b.empty:
        return
    show(b, "fade book-contracts team")
    dog = b.pick_is_dog.astype(str) == "True"
    show(b[dog], "  ...when the pick is the dog")
    show(b[~dog], "  ...when the pick is the fav")
    # baseline: the underdog in the same games, at the same DraftKings prices is not logged for
    # the other side, so compare cover rates: dog cover% in these games
    cov = np.where(dog, b.margin + b.pick_line > 0, b.margin + b.pick_line < 0)
    print(f"  baseline: underdog covered {100 * cov.mean():.1f}% of these games")


if __name__ == "__main__":
    raise SystemExit(main())
