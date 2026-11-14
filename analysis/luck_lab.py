"""Luck lab: which version of "luck" best predicts who fails to cover next?

  python analysis/luck_lab.py --data-dir store/data [--out reports/luck_lab.md]

Luck v1 (the locked rule) = actual margin - 129.5 x success-rate differential, averaged over
the season. It is noisy: one freak game or a 77-0 mismatch dominates the average, big plays
count as luck, and it treats every source of luck alike. This lab builds cleaner versions
and two penalty measures, all from games BEFORE the one being bet:

  v1        the locked rule's number
  capped    each game's luck clipped to +-CAP points (one blowout cannot own the average)
  two_fac   margin predicted from success rate AND yards per play (explosive offenses stop
            looking "lucky"); luck is what is left over
  clean     two_fac, clipped
  recent    clean, last 4 games only
  turnover  points from turnover margin alone (about 4.5 a turnover): the luck that repeats least
  close     wins minus expected wins in one-score games
  pen_share points on scoring drives credited to defensive-penalty yards
            (drive points x penalty yards / all yards gained on the drive), for minus against
  pen_ext   points on scoring drives kept alive by a defensive penalty on 3rd/4th down,
            for minus against

Each is graded two ways per season, betting AGAINST the luckier team at the closing spread:
  rank corr  correlation between the luck gap and the cover margin (negative = fading works),
             threshold-free, with a bootstrap interval
  top bucket ATS record when the gap is in the top 20% of that season's games
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
import explosive  # noqa: E402

_load, _games, _cache = explosive.load, bt.games, {}


def _cl(D, t):
    if t not in _cache:
        _cache[t] = _load(D, t)
    return _cache[t].copy()


def _cg(D):
    if "__g" not in _cache:
        _cache["__g"] = _games(D)
    return _cache["__g"].copy()


explosive.load, bt.games = _cl, _cg
import patterns as P  # noqa: E402

SLOPE, CAP, TO_PTS = 129.5, 17.0, 4.5
OUT: list[str] = []
RNG = np.random.default_rng(3)


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def drive_penalties(D: Path) -> pd.DataFrame:
    """Per game and offense: points on scoring drives credited to defensive penalties."""
    p = explosive.load(D, "plays").drop_duplicates(["espn_id", "play_id"])
    for c in ("seq", "down", "yards", "home_score", "away_score"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    g = P.espn_games(D).set_index("espn_id")
    p = p[p.espn_id.isin(g.index) & p.offense_id.notna()].sort_values(["espn_id", "seq"])
    typ = p.type.fillna("")
    p = p[~typ.isin(["Timeout", "End Period", "End of Half", "End of Game", "End of Regulation", "Coin Toss"])]
    typ = p.type.fillna("")
    kick = typ.str.startswith("Kickoff")
    # a drive: consecutive plays by the same offense; a kickoff always starts a new one
    new = (p.offense_id != p.groupby("espn_id").offense_id.shift()) | kick | kick.groupby(p.espn_id).shift(fill_value=False)
    p["drive"] = new.groupby(p.espn_id).cumsum()
    home = p.espn_id.map(g.home_id).astype(str)
    p["off_score"] = np.where(p.offense_id.astype(str) == home, p.home_score, p.away_score)
    pen = (typ == "Penalty") & (p.yards > 0)                      # flag on the defense
    p["pen_yds"] = np.where(pen, p.yards, 0.0)
    p["pen_ext"] = pen & p.down.isin([3, 4])
    scrim = ~kick & ~typ.isin(["Penalty", "Punt", "Punt Return"])
    p["gain"] = np.where(scrim, p.yards.clip(lower=0).fillna(0), 0.0)
    d = p[~kick].groupby(["espn_id", "drive", "offense_id"]).agg(
        s0=("off_score", "first"), s1=("off_score", "max"), pen=("pen_yds", "sum"), ext=("pen_ext", "any"),
        gain=("gain", "sum")).reset_index()
    d["pts"] = (d.s1 - d.s0).clip(0, 8)
    d["share"] = d.pts * d.pen / (d.pen + d.gain).replace(0, np.nan)
    d["ext_pts"] = np.where(d.ext, d.pts, 0.0)
    out = d.groupby(["espn_id", "offense_id"]).agg(pen_share=("share", "sum"), pen_ext=("ext_pts", "sum")).reset_index()
    return out.rename(columns={"offense_id": "team"}).assign(team=lambda x: x.team.astype(str))


def table(D: Path) -> pd.DataFrame:
    t = P.team_stats(D, slope=SLOPE)
    t["ypp_diff"] = t.ypp - t.ypp_against
    pen = drive_penalties(D)
    t = t.merge(pen, on=["espn_id", "team"], how="left")
    t = t.merge(pen.rename(columns={"team": "opp", "pen_share": "pen_share_a", "pen_ext": "pen_ext_a"}),
                on=["espn_id", "opp"], how="left")
    for c in ("pen_share", "pen_ext", "pen_share_a", "pen_ext_a"):
        t[c] = t[c].fillna(0.0)
    # two-factor expectation, fit on the 2025 season only and applied to both
    f = t[(t.season == 2025) & t.ypp_diff.notna()]
    X = np.c_[f.sr_diff, f.ypp_diff, np.ones(len(f))]
    b = np.linalg.lstsq(X, f.margin.clip(-45, 45), rcond=None)[0]
    say(f"- two-factor fit (2025): margin = {b[0]:.1f} x success-rate diff + {b[1]:.2f} x yards-per-play diff {b[2]:+.1f}")
    exp2 = b[0] * t.sr_diff + b[1] * t.ypp_diff.fillna(0)
    t["g_v1"] = t.luck
    t["g_capped"] = t.luck.clip(-CAP, CAP)
    t["g_two_fac"] = t.margin - exp2
    t["g_clean"] = (t.margin - exp2).clip(-CAP, CAP)
    t["g_turnover"] = TO_PTS * t.to_margin
    close = t.margin.abs() <= 8
    t["g_close"] = np.where(close, (t.margin > 0).astype(float) - 0.5, np.nan)
    t["g_pen_share"] = t.pen_share - t.pen_share_a
    t["g_pen_ext"] = t.pen_ext - t.pen_ext_a
    t = t.sort_values("kickoff")
    grp = t.groupby(["team", "season"])
    for v in ("v1", "capped", "two_fac", "clean", "turnover", "close", "pen_share", "pen_ext"):
        t[f"pre_{v}"] = grp[f"g_{v}"].transform(lambda s: s.shift().expanding().mean())
    t["pre_recent"] = grp.g_clean.transform(lambda s: s.shift().rolling(4, min_periods=3).mean())
    return t


VARIANTS = ["v1", "capped", "two_fac", "clean", "recent", "turnover", "close", "pen_share", "pen_ext"]


def gaps(D: Path, t: pd.DataFrame) -> pd.DataFrame:
    S = P.side_table(D)
    keep = ["espn_id", "side", "pre_n"] + [f"pre_{v}" for v in VARIANTS]
    me = t[keep]
    opp = me.assign(side=np.where(me.side == "home", "away", "home")).rename(
        columns={c: "o_" + c for c in keep if c not in ("espn_id", "side")})
    x = S.merge(me, on=["espn_id", "side"]).merge(opp, on=["espn_id", "side"])
    x = x[(x.pre_n >= 3) & (x.o_pre_n >= 3)]
    for v in VARIANTS:
        x[f"d_{v}"] = x[f"pre_{v}"] - x[f"o_pre_{v}"]             # + = this side has been luckier
    return x[x.side == "home"].copy()                             # one row a game: the gap is symmetric


def corr(d: pd.DataFrame, col: str) -> str:
    d = d[[col, "res"]].dropna()
    if len(d) < 50:
        return "n/a"
    r = d[col].rank().corr(d.res.rank())
    idx = RNG.integers(0, len(d), (800, len(d)))
    a, b = d[col].rank().values, d.res.rank().values
    bs = [np.corrcoef(a[i], b[i])[0, 1] for i in idx]
    return f"{r:+.3f} ({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f})"


def bucket(d: pd.DataFrame, col: str, q: float = 0.2) -> str:
    d = d[[col, "res", "odds"]].dropna(subset=[col, "res"])
    if len(d) < 50:
        return "n/a"
    hi, lo = d[col].quantile(1 - q / 2), d[col].quantile(q / 2)
    # fade the luckier team: home luckier -> away covers when res < 0; home unluckier -> home covers
    win = ((d[col] >= hi) & (d.res < 0)) | ((d[col] <= lo) & (d.res > 0))
    loss = ((d[col] >= hi) & (d.res > 0)) | ((d[col] <= lo) & (d.res < 0))
    w, l_ = int(win.sum()), int(loss.sum())
    return f"{w}-{l_} ({100 * w / max(w + l_, 1):.1f}%)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    D = Path(a.data_dir)
    say(f"# Luck lab ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d})")
    t = table(D)
    say(f"- team-games {len(t)} | avg points credited to defensive penalties per team-game: share method "
        f"{t.pen_share.mean():.2f}, drive-extending method {t.pen_ext.mean():.2f}")
    x = gaps(D, t)
    say(f"- games with both teams at 3+ prior games and a closing spread: {x.groupby('season').size().to_dict()}")
    for scope, d in (("All games", x), ("FCS", x[x.division == "FCS"]), ("FBS", x[x.division == "FBS"])):
        say(f"\n## {scope}: fading the luckier team")
        say("| version | 2025 rank corr (95%) | 2026 rank corr (95%) | 2025 top-20% gaps | 2026 top-20% gaps |")
        say("|---|---|---|---|---|")
        for v in VARIANTS:
            c = f"d_{v}"
            a5, a6 = d[d.season == 2025], d[d.season == 2026]
            say(f"| {v} | {corr(a5, c)} | {corr(a6, c)} | {bucket(a5, c)} | {bucket(a6, c)} |")
    say("\n- read: a NEGATIVE correlation means the luckier team covers less (fading works). A version is better "
        "than v1 only if it is more negative in BOTH seasons.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(OUT) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
