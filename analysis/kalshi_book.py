"""Does Kalshi's money side agree with the line move, and what happens betting with or against it?

  python analysis/kalshi_book.py --data-dir store/data

Two "where is the money" signals per game, both as of kickoff:
  book   resting Kalshi order book: $ (price x size) of bids backing each team in the
         last snapshot before kickoff (only collected live, so 2026 only)
  traded Kalshi taker dollars bought on each team, market open -> kickoff (both seasons)
"Money team" = the team with more dollars. Every bet is priced at the closing
DraftKings line/odds (consensus if missing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import backtest as bt  # noqa: E402
from ncaaf_tracker import analyze as an  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402


def signals(D: Path) -> pd.DataFrame:
    games = an.build_games(D)
    mt = Matcher(games)
    snaps, flows = an.build_kalshi(D, games, mt, [])
    ko = games.set_index("espn_id").kickoff_dt
    out = pd.DataFrame(index=pd.Index(games.espn_id, name="espn_id"))
    if "kalshi_book_imbalance" in snaps:
        b = snaps.dropna(subset=["kalshi_book_imbalance"])
        b = b[b.snap < b.espn_id.map(ko)].sort_values("snap").groupby("espn_id").last()
        out["book"] = b.kalshi_book_imbalance
        out["book_hrs"] = (b.index.to_series().map(ko) - b.snap).dt.total_seconds() / 3600
    if not flows.empty:
        f = flows[flows.hour < flows.espn_id.map(ko)]
        out["traded"] = f.groupby("espn_id").kalshi_flow.sum()
    return out.reset_index()


def line(d: pd.DataFrame, pick_home: pd.Series, label: str) -> None:
    r = bt.score(d.reset_index(drop=True), pd.Series(pick_home.values, dtype=object))
    print(f"  {label:44} n={r['picks']:4d}  ATS {r['cover%']:4.1f}% ROI {r['ATS ROI%']:+6.1f}% | "
          f"ML win {r['win%']:4.1f}% ROI {r['ML ROI%']:+6.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    D = Path(ap.parse_args(argv).data_dir)

    df = bt.games(D).merge(signals(D), on="espn_id", how="left")
    df = df[df.sp_close.notna()]
    move = df.sp_close - df.sp_open                       # < 0: line moved toward home
    df["move_side"] = np.where(move <= -1, 1, np.where(move >= 1, -1, 0))
    print(f"games with closing spread {len(df)} | book signal {df.book.notna().sum()} "
          f"(median {df.book_hrs.median():.1f}h before kickoff) | traded signal {df.traded.notna().sum()}")

    for sig in ("book", "traded"):
        for scope, sm in (("all games", df.n_ranked >= 0), ("any ranked", df.n_ranked >= 1),
                          ("ranked vs ranked", df.n_ranked == 2)):
            d = df[sm & df[sig].notna() & (df[sig] != 0)]
            if d.empty:
                continue
            mv = d[d.move_side != 0]
            agree = (np.sign(mv[sig]) == mv.move_side)
            print(f"\n{sig.upper()} money, {scope}: money side == line-move side in {agree.sum()}/{len(mv)} "
                  f"({100 * agree.mean():.0f}%) of games where the line moved 1+")
            for season in [*sorted(d.season.unique()), None]:
                s = d if season is None else d[d.season == season]
                tag = season or "both"
                h = s[sig] > 0
                line(s, h, f"{tag}: back the money team")
                line(s, ~h, f"{tag}: FADE the money team")
            for name, cond in (("line moved WITH the money", np.sign(d[sig]) == d.move_side),
                               ("line moved AGAINST the money", np.sign(d[sig]) == -d.move_side)):
                s = d[cond]
                line(s, ~(s[sig] > 0), f"FADE money team, {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
