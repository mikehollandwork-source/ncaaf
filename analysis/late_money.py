"""Money and order counts on each team in the last N minutes before kickoff, Kalshi and
Polymarket (game-winner markets), and how those sides did.

  python analysis/late_money.py --data-dir store/data [--minutes 10] [--season 2026] [--out games.csv]

"Orders" are executed trades (fills): the order book is only snapshotted when the collector
runs, so resting orders at exactly T-10 aren't available. A trade backs a team when the taker
buys that team (Kalshi: yes on its ticker or no on the opponent's; Polymarket: BUY its token
or SELL the opponent's). Dollars = what the taker paid. Results graded at the closing
DraftKings spread/moneyline (consensus if missing).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import backtest as bt  # noqa: E402
from ncaaf_tracker import analyze as an  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402


def read(D: Path, table: str, cols: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(D / table / "*" / "*.csv.gz")))
    return pd.concat((pd.read_csv(f, usecols=cols, dtype=str) for f in files), ignore_index=True)


def kalshi(D: Path, games: pd.DataFrame, mt: Matcher, lo: pd.Timestamp) -> pd.DataFrame:
    m = an.load(D, "kalshi_markets")
    m = m[m.series == "KXNCAAFGAME"]
    tick = an.kalshi_ticker_sides(m, mt, [])
    t = read(D, "kalshi_trades", ["trade_id", "ticker", "created_time", "yes_price", "no_price", "count", "taker_side"])
    t = t[t.ticker.isin(tick) & (t.created_time >= lo.isoformat())].drop_duplicates("trade_id")
    an.num(t, ["yes_price", "no_price", "count"])
    t["espn_id"] = t.ticker.map(lambda x: tick[x][0])
    team = t.ticker.map(lambda x: tick[x][1])
    t["backs"] = np.where(t.taker_side == "yes", team, np.where(team == "home", "away", "home"))
    t["usd"] = np.where(t.taker_side == "yes", t.yes_price, t.no_price) * t["count"]
    t["ts"] = an.ts(t.created_time)
    return t[["espn_id", "backs", "usd", "ts"]]


def poly(D: Path, games: pd.DataFrame, mt: Matcher, lo: pd.Timestamp) -> pd.DataFrame:
    m = an.load(D, "poly_markets")
    m = m[m.sports_market_type == "moneyline"].drop_duplicates("condition_id", keep="last")
    tok: dict[str, tuple[str, str]] = {}
    for _, r in m.iterrows():
        if pd.isna(r.outcome_0) or pd.isna(r.outcome_1):
            continue
        gid, sides = mt.match(r.outcome_0, r.outcome_1, pd.Timestamp(r.game_start_time), 12)
        if gid is None:
            continue
        tok[str(r.token_0)] = (gid, sides[r.outcome_0])
        tok[str(r.token_1)] = (gid, sides[r.outcome_1])
    t = read(D, "poly_trades", ["trade_id", "token_id", "side", "price", "size", "trade_ts"])
    t = t[t.token_id.isin(tok) & (t.trade_ts >= lo.isoformat())].drop_duplicates("trade_id")
    an.num(t, ["price", "size"])
    t["espn_id"] = t.token_id.map(lambda x: tok[x][0])
    team = t.token_id.map(lambda x: tok[x][1])
    buy = t.side.str.upper() == "BUY"
    t["backs"] = np.where(buy, team, np.where(team == "home", "away", "home"))
    t["usd"] = np.where(buy, t.price, 1 - t.price) * t["size"]
    t["ts"] = an.ts(t.trade_ts)
    return t[["espn_id", "backs", "usd", "ts"]]


def window(t: pd.DataFrame, ko: pd.Series, minutes: int, tag: str) -> pd.DataFrame:
    k = t.espn_id.map(ko)
    w = t[(t.ts < k) & (t.ts >= k - pd.Timedelta(minutes=minutes))]
    a = w.pivot_table(index="espn_id", columns="backs", values="usd", aggfunc=["sum", "count"], fill_value=0)
    a.columns = [f"{tag}_{'usd' if f == 'sum' else 'n'}_{s}" for f, s in a.columns]
    for c in (f"{tag}_usd_home", f"{tag}_usd_away", f"{tag}_n_home", f"{tag}_n_away"):
        if c not in a:
            a[c] = 0
    # coverage: did we capture any trade for this game after the window opened at all?
    last = t.groupby("espn_id").ts.max()
    a = a.reindex(ko.index)
    a[f"{tag}_covered"] = (last.reindex(ko.index) >= ko - pd.Timedelta(minutes=minutes)).fillna(False)
    return a


def rec(d: pd.DataFrame, pick_home: pd.Series, label: str) -> None:
    d = d[pick_home.notna()]
    if d.empty:
        return
    r = bt.score(d.reset_index(drop=True), pd.Series(pick_home[pick_home.notna()].values, dtype=object))
    print(f"  {label:52} n={r['picks']:3d}  ATS {r['cover%']:5.1f}% ROI {r['ATS ROI%']:+6.1f}% | "
          f"ML win {r['win%']:5.1f}% ROI {r['ML ROI%']:+6.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    D = Path(args.data_dir)
    pd.set_option("display.width", 250)

    games = an.build_games(D)
    mt = Matcher(games)
    g = bt.games(D)
    g = g[(g.season == args.season) & g.sp_close.notna()].copy()
    ko = games.set_index("espn_id").kickoff_dt.reindex(g.espn_id)
    lo = ko.min() - pd.Timedelta(minutes=args.minutes)
    df = g.set_index("espn_id")
    for tag, fn in (("k", kalshi), ("p", poly)):
        df = df.join(window(fn(D, games, mt, lo), ko, args.minutes, tag))
    df = df.reset_index()

    M = args.minutes
    print(f"{args.season} completed games with a closing spread: {len(df)}")
    for tag, name in (("k", "Kalshi"), ("p", "Polymarket")):
        tot_usd = df[f"{tag}_usd_home"] + df[f"{tag}_usd_away"]
        tot_n = df[f"{tag}_n_home"] + df[f"{tag}_n_away"]
        cov = df[f"{tag}_covered"].fillna(False).astype(bool)
        has = cov & (tot_n > 0)
        print(f"\n{name}: last-{M}-min window captured for {cov.sum()} games, with trades in {has.sum()}")
        if has.any():
            print(f"  per game in the window: median ${tot_usd[has].median():,.0f} over {tot_n[has].median():.0f} trades "
                  f"| mean ${tot_usd[has].mean():,.0f} | biggest ${tot_usd[has].max():,.0f}")
        df[f"{tag}_has"] = has
        df[f"{tag}_usd_side"] = np.where(~has, np.nan, np.sign(df[f"{tag}_usd_home"] - df[f"{tag}_usd_away"]))
        df[f"{tag}_n_side"] = np.where(~has, np.nan, np.sign(df[f"{tag}_n_home"] - df[f"{tag}_n_away"]))
        df[f"{tag}_usd_share"] = df[f"{tag}_usd_home"] / tot_usd.where(tot_usd > 0)

    fav_home = df.sp_close < 0
    pick = lambda cond, home: pd.Series(np.where(cond, home, np.nan), index=df.index, dtype=object)
    for tag, name in (("k", "Kalshi"), ("p", "Polymarket")):
        h = df[f"{tag}_has"]
        if not h.any():
            continue
        us, ns = df[f"{tag}_usd_side"], df[f"{tag}_n_side"]
        print(f"\n{name}, last {M} min before kickoff:")
        print(f"  more $ on the favorite in {100 * ((us > 0) == fav_home)[h & (us != 0)].mean():.0f}% of games; "
              f"$ side and order-count side agree in {100 * (us == ns)[h & (us != 0) & (ns != 0)].mean():.0f}%")
        rec(df, pick(h & (us != 0), us > 0), "back the team with more $")
        rec(df, pick(h & (us != 0), us < 0), "fade the team with more $")
        rec(df, pick(h & (ns != 0), ns > 0), "back the team with more orders")
        rec(df, pick(h & (ns != 0), ns < 0), "fade the team with more orders")
        split = h & (us != 0) & (ns != 0) & (us != ns)
        rec(df, pick(split, us > 0), "$ and orders disagree: back the $ side (bigger bets)")
        rec(df, pick(split, ns > 0), "$ and orders disagree: back the orders side (more bets)")
        share = df[f"{tag}_usd_share"]
        big = h & ((share >= 0.8) | (share <= 0.2))
        rec(df, pick(big, share >= 0.8), "lopsided: 80%+ of the window $ on one team, back it")
        rec(df, pick(big, share < 0.5), "lopsided: 80%+ of the window $ on one team, fade it")
    both = df.k_has & df.p_has & (df.k_usd_side != 0) & (df.p_usd_side != 0)
    if both.any():
        agree = both & (df.k_usd_side == df.p_usd_side)
        print(f"\nKalshi and Polymarket late $ on the same team in {agree.sum()}/{both.sum()} games")
        rec(df, pick(agree, df.k_usd_side > 0), "both markets' late $ on one team, back it")
        rec(df, pick(both & ~agree, df.k_usd_side > 0), "they disagree: back Kalshi's side")

    cols = ["kickoff", "away", "home", "sp_close", "hs", "as", "k_usd_away", "k_usd_home", "k_n_away", "k_n_home",
            "p_usd_away", "p_usd_home", "p_n_away", "p_n_home"]
    show = df[df.k_has | df.p_has].copy()
    show["late_total"] = show[["k_usd_away", "k_usd_home", "p_usd_away", "p_usd_home"]].sum(axis=1)
    show["kickoff"] = show.kickoff.dt.strftime("%m-%d %H:%M")
    print(f"\nBiggest last-{M}-minute money games:")
    print(show.nlargest(15, "late_total")[cols].round(0).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
