"""Brute-force edge search: every 1-, 2- and 3-way combination of the signals we collect,
checked for an against-the-spread / moneyline edge that holds in BOTH seasons.

  python analysis/edge_search.py --data-dir store/data [--kalshi signals.csv] [--perms 50]

One row per team per game (the "side"). Signals, all from that side's point of view:
  sp_move   spread move open -> DraftKings close toward the side (pts)
  ml_move   no-vig moneyline win chance change open -> close (pct pts)
  sp_tix / sp_mon / sp_gap   spread tickets %, money %, money - tickets ("sharp" gap)
  ml_tix / ml_gap            moneyline tickets %, money - tickets
  fav, home, size, rank, div  closing favourite/dog, home/away, |spread|, AP ranks, FBS/FCS
  kalshi    Kalshi taker money on the side (with/against), open -> kickoff (--kalshi cache)

Guard rails against data mining: a pattern must clear the bar in 2025 AND 2026, each with
enough bets, and the same search is rerun on shuffled outcomes to count how many such
"edges" pure luck produces.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import bands  # noqa: E402
import backtest as bt  # noqa: E402
import explosive  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402

MIN_N = 30          # bets per season
BAR = 55.0          # cover % needed in each season (break-even at -110 is 52.4%)


def splits(D: Path) -> pd.DataFrame:
    """Action Network consensus tickets/money per ESPN game and side, spreads + moneyline."""
    fl = explosive.load(D, "an_final_lines")
    fl = fl[(fl.book == "an_consensus") & fl.market.isin(["spreads", "h2h"])].copy()
    for c in ("money_pct", "tickets_pct"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce")
    fl.loc[~fl.money_pct.between(3, 96), ["money_pct", "tickets_pct"]] = np.nan   # bogus 0/100 rows
    g = explosive.load(D, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last().reset_index()
    m = Matcher(g)
    rows = []
    for _, e in fl.groupby("event_id"):
        r = e.iloc[0]
        gid, s = m.match(r.home_team, r.away_team, pd.Timestamp(r.commence_time), 12)
        if gid is None:
            continue
        for _, o in e.iterrows():
            side = "home" if (o.side == "home") == (s[r.home_team] == "home") else "away"
            rows.append((gid, side, o.market, o.tickets_pct, o.money_pct))
    d = pd.DataFrame(rows, columns=["espn_id", "side", "market", "tix", "mon"])
    d = d.drop_duplicates(["espn_id", "side", "market"], keep="last")
    p = d.pivot_table(index=["espn_id", "side"], columns="market", values=["tix", "mon"])
    p.columns = [f"{'sp' if mk == 'spreads' else 'ml'}_{v}" for v, mk in p.columns]
    return p.reset_index()


def nv(h, a):
    if np.isnan(h) or np.isnan(a):
        return np.nan
    ph, pa = bands.implied(h), bands.implied(a)
    return ph / (ph + pa)


def side_rows(D: Path, kalshi: str | None) -> pd.DataFrame:
    g = bt.games(D)
    g = g[g.sp_close.notna() & g.sp_open.notna()].copy()
    g["p_open"] = [nv(h, a) for h, a in zip(g.ml_h_open, g.ml_a_open)]
    g["p_close"] = [nv(h, a) for h, a in zip(g.ml_h_close, g.ml_a_close)]
    if kalshi:
        k = pd.read_csv(kalshi, dtype={"espn_id": str})[["espn_id", "traded"]]
        g = g.merge(k, on="espn_id", how="left")
    else:
        g["traded"] = np.nan
    out = []
    for side, sgn in (("home", 1), ("away", -1)):
        margin = sgn * (g.hs - g["as"])
        line = g.sp_close if side == "home" else -g.sp_close
        own = (g.home_rank if side == "home" else g.away_rank).notna()
        opp = (g.away_rank if side == "home" else g.home_rank).notna()
        out.append(pd.DataFrame({
            "espn_id": g.espn_id, "season": g.season, "side": side,
            "res": margin + line,
            "sp_odds": g.sp_h_odds if side == "home" else g.sp_a_odds,
            "won": margin > 0,
            "ml_odds": g.ml_h_close if side == "home" else g.ml_a_close,
            "sp_move_v": -sgn * (g.sp_close - g.sp_open),
            "ml_move_v": 100 * sgn * (g.p_close - g.p_open),
            "line": line, "own_rk": own, "opp_rk": opp, "division": g.division,
            "kal_v": sgn * g.traded,
        }))
    s = pd.concat(out, ignore_index=True).merge(splits(D), on=["espn_id", "side"], how="left")
    cut = lambda v, e, lab: pd.cut(v, e, labels=lab).astype(object)
    s["sp_move"] = cut(s.sp_move_v, [-99, -1.75, -0.75, 0.75, 1.75, 99], ["away2+", "away1", "flat", "toward1", "toward2+"])
    s["ml_move"] = cut(s.ml_move_v, [-99, -4, -1.5, 1.5, 4, 99], ["fell4+", "fell", "flat", "rose", "rose4+"])
    gap_sp, gap_ml = s.sp_mon - s.sp_tix, s.ml_mon - s.ml_tix
    pct = [-1, 30, 45, 55, 70, 101]
    plab = ["<30", "30-45", "45-55", "55-70", "70+"]
    s["sp_tix"] = cut(s.sp_tix, pct, plab)
    s["sp_mon"] = cut(s.sp_mon, pct, plab)
    s["sp_gap"] = cut(gap_sp, [-101, -10, -3, 3, 10, 101], ["m<<t", "m<t", "even", "m>t", "m>>t"])
    s["ml_tix"] = cut(s.ml_tix, pct, plab)
    s["ml_gap"] = cut(gap_ml, [-101, -10, -3, 3, 10, 101], ["m<<t", "m<t", "even", "m>t", "m>>t"])
    s["fav"] = np.where(s.line < 0, "fav", np.where(s.line > 0, "dog", None))
    s["home"] = s.side
    s["size"] = cut(s.line.abs(), [-1, 3.25, 7.25, 14.25, 99], ["0-3", "3.5-7", "7.5-14", "14.5+"])
    s["rank"] = np.select([s.own_rk & s.opp_rk, s.own_rk, s.opp_rk], ["both", "side", "opp"], "none")
    s["div"] = s.division
    s["kalshi"] = np.where(s.kal_v.isna() | (s.kal_v == 0), None, np.where(s.kal_v > 0, "with", "against"))
    s["cover"] = np.sign(s.res)
    s["sp_p"] = np.where(s.res == 0, 0.0, np.where(s.res > 0, [bands.american_profit(-110 if np.isnan(o) else o)
                                                                 for o in s.sp_odds], -1.0))
    s["ml_p"] = np.where(s.ml_odds.isna(), np.nan,
                         np.where(s.won, [bands.american_profit(o) if not np.isnan(o) else np.nan for o in s.ml_odds], -1.0))
    return s


FEATURES = ["sp_move", "ml_move", "sp_tix", "sp_mon", "sp_gap", "ml_tix", "ml_gap", "fav", "home", "size",
            "rank", "div", "kalshi"]


def scan(s: pd.DataFrame, cover: np.ndarray, max_k: int = 3, with_roi: bool = True) -> pd.DataFrame:
    s = s.assign(w=(cover > 0).astype(int), l=(cover < 0).astype(int))
    rows = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(FEATURES, k):
            d = s.dropna(subset=list(combo))
            if d.empty:
                continue
            if with_roi:
                a = d.groupby(["season", *combo]).agg(w=("w", "sum"), l=("l", "sum"), sp_p=("sp_p", "sum"),
                                                      ml_p=("ml_p", "sum"), ml_n=("ml_p", "count"))
            else:
                a = d.groupby(["season", *combo]).agg(w=("w", "sum"), l=("l", "sum"))
            a = a.reset_index()
            a["combo"] = a[list(combo)].astype(str).agg(" & ".join, axis=1)
            a["features"] = " & ".join(combo)
            rows.append(a.drop(columns=list(combo)))
    r = pd.concat(rows, ignore_index=True)
    w = r.pivot_table(index=["features", "combo"], columns="season",
                      values=[c for c in ("w", "l", "sp_p", "ml_p", "ml_n") if c in r], fill_value=0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index()


def survivors(w: pd.DataFrame, s1=2025, s2=2026) -> pd.DataFrame:
    c = {}
    for y in (s1, s2):
        n = w[f"w_{y}"] + w[f"l_{y}"]
        c[y] = (n >= MIN_N) & (100 * w[f"w_{y}"] / n.where(n > 0) >= BAR)
    return w[c[s1] & c[s2]]


def best_min(w: pd.DataFrame) -> float:
    """Best pattern by its WORSE season's cover % (n >= MIN_N in both seasons)."""
    n1, n2 = w.w_2025 + w.l_2025, w.w_2026 + w.l_2026
    ok = (n1 >= MIN_N) & (n2 >= MIN_N)
    return float(np.minimum(100 * w.w_2025 / n1, 100 * w.w_2026 / n2)[ok].max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--kalshi")
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    s = side_rows(Path(args.data_dir), args.kalshi)
    print(f"sides {len(s)} | games by season {s.drop_duplicates('espn_id').season.value_counts().to_dict()}")
    for f in FEATURES:
        print(f"  {f:8} coverage {100 * s[f].notna().mean():3.0f}%  values {sorted(s[f].dropna().unique())}")
    w = scan(s, s.cover.values)
    print(f"\nconditions tested: {len(w)}")
    real = survivors(w)

    # null: flip each game's cover result at random (keeps the two sides of a game mirrored)
    rng = np.random.default_rng(7)
    games = s.espn_id.unique()
    idx = pd.Index(games)
    null_counts, null_best = [], []
    for _ in range(args.perms):
        flip = pd.Series(rng.choice([-1, 1], len(games)), index=idx)
        nw = scan(s, s.cover.values * s.espn_id.map(flip).values, with_roi=False)
        null_counts.append(len(survivors(nw)))
        null_best.append(best_min(nw))
    nc, nb = np.array(null_counts), np.array(null_best)
    print(f"best pattern's worse-season cover: real {best_min(w):.1f}% | shuffled: mean {nb.mean():.1f}%, "
          f"95th pct {np.percentile(nb, 95):.1f}% -> share of shuffles at least as good: "
          f"{100 * (nb >= best_min(w)).mean():.0f}%")
    print(f"patterns >= {BAR}% ATS with n >= {MIN_N} in BOTH seasons: real {len(real)} | "
          f"shuffled outcomes: mean {nc.mean():.1f}, 95th pct {np.percentile(nc, 95):.0f}, max {nc.max()}")

    def fmt(t: pd.DataFrame) -> pd.DataFrame:
        o = pd.DataFrame({"pattern": t.combo, "signals": t.features})
        for y in (2025, 2026):
            n = t[f"w_{y}"] + t[f"l_{y}"]
            o[f"{y}"] = (t[f"w_{y}"].astype(int).astype(str) + "-" + t[f"l_{y}"].astype(int).astype(str)
                         + " (" + (100 * t[f"w_{y}"] / n).round(1).astype(str) + "%)")
            o[f"atsROI{y}"] = (100 * t[f"sp_p_{y}"] / n).round(1)
            o[f"mlROI{y}"] = (100 * t[f"ml_p_{y}"] / t[f"ml_n_{y}"].where(t[f"ml_n_{y}"] > 0)).round(1)
        o["n"] = (t.w_2025 + t.l_2025 + t.w_2026 + t.l_2026).astype(int)
        o["both%"] = (100 * (t.w_2025 + t.w_2026) / o.n).round(1)
        return o.sort_values("both%", ascending=False)

    print("\nSurvivors (both seasons):")
    print(fmt(real).to_string(index=False))

    # out-of-sample view: best 2025 patterns -> how they did in 2026, and vice versa
    for a, b in ((2025, 2026), (2026, 2025)):
        n_a = w[f"w_{a}"] + w[f"l_{a}"]
        n_b = w[f"w_{b}"] + w[f"l_{b}"]
        top = w[(n_a >= 50) & (n_b >= 20)].assign(p=100 * w[f"w_{a}"] / n_a).nlargest(25, "p")
        oos = 100 * top[f"w_{b}"].sum() / (top[f"w_{b}"] + top[f"l_{b}"]).sum()
        print(f"\nTop 25 patterns found in {a} (n>=50) -> combined cover in {b}: {oos:.1f}% "
              f"(they averaged {top.p.mean():.1f}% in {a})")
    if args.out:
        fmt(w).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
