"""Fade the Kalshi order-book money team: where it wins and loses, and how it lines up
with Action Network's public tickets and money.

  python analysis/book_fade.py --data-dir store/data [--out games.csv]

Book side per game = last order-book snapshot before kickoff. "Money team" = the team with
more resting bid dollars (price x contracts); --by contracts uses contract counts instead,
which removes the tilt toward favorites (their contracts cost more). Bets are graded at the
closing DraftKings line/odds (consensus if missing). The order book is only captured live,
so this is 2026 data only.
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
import money_agreement as ma  # noqa: E402
from ncaaf_tracker import analyze as an  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402


def book_sides(D: Path) -> pd.DataFrame:
    games = an.build_games(D)
    m = an.load(D, "kalshi_markets")
    m = m[m["series"] == "KXNCAAFGAME"].copy()
    tick = an.kalshi_ticker_sides(m, Matcher(games), [])
    b = an.load(D, "kalshi_books")
    an.num(b, ["price", "size"])
    b = b[b.ticker.isin(tick)].copy()
    b["snap"] = an.ts(b.snapshot_ts)
    b["espn_id"] = b.ticker.map(lambda t: tick[t][0])
    team = b.ticker.map(lambda t: tick[t][1])
    b["backs"] = np.where(b.side == "yes", team, np.where(team == "home", "away", "home"))
    b["usd"] = b.price * b["size"]
    # only snapshots that caught both teams' books
    both = b.groupby(["espn_id", "snap"]).ticker.nunique() == 2
    b = b.set_index(["espn_id", "snap"]).loc[both[both].index].reset_index()
    p = b.pivot_table(index=["espn_id", "snap"], columns="backs", values=["usd", "size"],
                      aggfunc="sum", fill_value=0)
    p.columns = [f"{c}_{a}" for a, c in p.columns]
    p = p.reset_index()
    ko = games.set_index("espn_id").kickoff_dt
    p = p[p.snap < p.espn_id.map(ko)].sort_values("snap").groupby("espn_id").last().reset_index()
    p["hrs_before"] = (p.espn_id.map(ko) - p.snap).dt.total_seconds() / 3600
    # Kalshi home win price at that snapshot (mid of the home ticker, else 1 - away mid)
    an.num(m, ["yes_bid", "yes_ask", "last_price"])
    m = m[m.ticker.isin(tick)].copy()
    m["snap"] = an.ts(m.snapshot_ts)
    m["espn_id"] = m.ticker.map(lambda t: tick[t][0])
    m["home_p"] = an.mid(m.yes_bid, m.yes_ask, m.last_price)
    m.loc[m.ticker.map(lambda t: tick[t][1]) == "away", "home_p"] = 1 - m.home_p
    kp = m.groupby(["espn_id", "snap"]).home_p.mean().rename("kalshi_home_p").reset_index()
    return p.merge(kp, on=["espn_id", "snap"], how="left")


def american(p: float) -> float:
    return np.nan if pd.isna(p) else (-100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p)


def rec(d: pd.DataFrame, label: str) -> None:
    if d.empty:
        return
    r = bt.score(d.reset_index(drop=True), pd.Series(d.fade_home.values, dtype=object))
    w = int(((d.ats > 0)).sum()); l_ = int((d.ats < 0).sum())
    print(f"  {label:34} n={r['picks']:3d}  ATS {w:3d}-{l_:<3d} {r['cover%']:5.1f}% ROI {r['ATS ROI%']:+6.1f}% | "
          f"ML {int(d.won.sum()):3d}-{int((~d.won).sum()):<3d} ROI {r['ML ROI%']:+6.1f}%")


def bands(d: pd.DataFrame, col: str, edges: list, labels: list, title: str) -> None:
    print(f"\n{title}")
    cut = pd.cut(d[col], edges, labels=labels)
    for lab in labels:
        rec(d[cut == lab], str(lab))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--by", choices=("usd", "size"), default="usd")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    D = Path(args.data_dir)

    g = bt.games(D)
    df = g.merge(book_sides(D), on="espn_id")
    df = df[df.sp_close.notna()].copy()
    by = args.by
    df["imb"] = (df[f"home_{by}"] - df[f"away_{by}"]) / (df[f"home_{by}"] + df[f"away_{by}"])
    df["imb_usd"] = (df.home_usd - df.away_usd) / (df.home_usd + df.away_usd)
    df["imb_size"] = (df.home_size - df.away_size) / (df.home_size + df.away_size)
    df = df[df.imb.notna() & (df.imb != 0)].copy()
    df["fade_home"] = df.imb < 0                       # fade = team with LESS book money
    df["book_total"] = df.home_usd + df.away_usd
    h = df.fade_home
    margin = df.hs - df["as"]
    df["ats"] = np.where(h, margin + df.sp_close, -(margin + df.sp_close))
    df["won"] = np.where(h, margin > 0, margin < 0)
    df["fade_ml"] = np.where(h, df.ml_h_close, df.ml_a_close)
    df["fade_spread"] = np.where(h, df.sp_close, -df.sp_close)
    df["fade_kalshi_p"] = np.where(h, df.kalshi_home_p, 1 - df.kalshi_home_p)
    move = df.sp_close - df.sp_open                    # < 0: moved toward home
    df["fade_move"] = np.where(h, -move, move)         # > 0: line moved toward the fade team
    df["strength"] = df.imb.abs()

    sp = ma.splits(D, set(df.espn_id))
    sp = sp[sp.money.between(3, 96)]
    sp = sp.pivot_table(index="espn_id", columns="backed", values=["tickets", "money"])
    sp.columns = [f"{a}_{b}" for a, b in sp.columns]
    df = df.merge(sp.reset_index(), on="espn_id", how="left")
    df["fade_tickets"] = np.where(df.fade_home, df.get("tickets_home"), df.get("tickets_away"))
    df["fade_money"] = np.where(df.fade_home, df.get("money_home"), df.get("money_away"))

    print(f"games {len(df)} (book measured in {by}) | snapshot median {df.hrs_before.median():.1f}h "
          f"before kickoff | with AN splits {df.fade_tickets.notna().sum()}")
    print(f"book-money team is the closing favorite in {100 * (df.fade_spread > 0).mean():.0f}% of games "
          f"(USD) — contracts-based: {100 * ((df.imb_size > 0) == (df.sp_close < 0)).mean():.0f}% "
          f"| USD and contracts pick the same team in {100 * (np.sign(df.imb_usd) == np.sign(df.imb_size)).mean():.0f}%")
    rec(df, "FADE the book money team (all)")

    bands(df, "fade_ml", [-1e9, -300, -150, -100, 150, 300, 1e9],
          ["fade fav -300+", "fade fav -150..-300", "fade fav -100..-150", "fade dog +100..+150",
           "fade dog +150..+300", "fade dog +300+"], "By the fade team's closing moneyline:")
    bands(df, "fade_spread", [-99, -14.25, -7.25, -3.25, 0, 3.25, 7.25, 14.25, 99],
          ["fav by 14.5+", "fav by 7.5-14", "fav by 3.5-7", "fav by 0.5-3", "dog 0.5-3", "dog 3.5-7",
           "dog 7.5-14", "dog 14.5+"], "By the fade team's closing spread:")
    bands(df, "fade_kalshi_p", [0, .2, .35, .5, .65, .8, 1],
          ["Kalshi <20c", "20-35c", "35-50c", "50-65c", "65-80c", "80c+"], "By the fade team's Kalshi price:")
    bands(df, "strength", [0, .2, .4, .6, .8, 1.0001],
          ["book lean 0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], "By how lopsided the book is:")
    bands(df, "book_total", [0, 1000, 5000, 20000, 1e12],
          ["book < $1k", "$1k-5k", "$5k-20k", "$20k+"], "By total resting book $:")
    bands(df, "hrs_before", [0, 1, 3, 12, 1e4],
          ["< 1h before", "1-3h", "3-12h", "12h+"], "By snapshot time before kickoff:")
    bands(df, "fade_move", [-99, -0.75, 0.75, 99],
          ["line moved AWAY from fade team", "line flat (<1)", "line moved TOWARD fade team"],
          "By line move (open -> DK close):")
    print("\nBy rank:")
    rec(df[df.n_ranked == 0], "no ranked team")
    rec(df[df.n_ranked >= 1], "any ranked")

    s = df.dropna(subset=["fade_tickets"])
    print("\nPublic tickets / money (Action Network) vs Kalshi book, from the FADE team's side:")
    bands(s, "fade_tickets", [0, 30, 40, 50, 60, 70, 100],
          ["fade tickets <30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"], "  by tickets % on the fade team:")
    pub_with = s.fade_tickets < 50          # public tickets also on the book money team
    print(f"\n  public tickets majority on the same team as Kalshi book money: {pub_with.sum()}/{len(s)} "
          f"({100 * pub_with.mean():.0f}%)")
    rec(s[pub_with], "agree: tickets + book same team")
    rec(s[~pub_with], "disagree: tickets on the fade team")
    mw = s.fade_money < 50
    print(f"  public money majority on the same team as Kalshi book money: {mw.sum()}/{len(s)}")
    rec(s[pub_with & mw], "tickets AND money with the book")
    rec(s[~pub_with & ~mw], "tickets AND money on the fade team")
    rec(s[pub_with & ~mw], "tickets with book, money on fade")
    rec(s[~pub_with & mw], "money with book, tickets on fade")
    print("\n  tickets% minus Kalshi book% on the book-money team (positive = public heavier than Kalshi):")
    book_pct = 50 + 50 * s.strength
    gap = (100 - s.fade_tickets) - book_pct
    for lab, c in (("public much lighter (gap < -25)", gap < -25), ("-25..0", (gap >= -25) & (gap < 0)),
                   ("0..+25", (gap >= 0) & (gap < 25)), ("public much heavier (> +25)", gap >= 25)):
        rec(s[c], lab)

    # Price-neutral version: in a matched market a team's backers put up its price per contract
    # and the other side 1 - price, so its fair share of book dollars ~= its Kalshi price.
    # excess = USD share - Kalshi price; > 0 means more resting money than the price implies.
    x = df[df.kalshi_home_p.between(0.02, 0.98)].copy()
    mg = x.hs - x["as"]
    x["excess"] = x.home_usd / (x.home_usd + x.away_usd) - x.kalshi_home_p
    x["fade_home"] = x.excess < 0                       # rec() grades fade_home; flip to back
    x["ats"] = np.where(x.fade_home, mg + x.sp_close, -(mg + x.sp_close))
    x["won"] = np.where(x.fade_home, x.hs > x["as"], x["as"] > x.hs)
    fav = (x.excess > 0) == (x.sp_close < 0)            # excess-demand team is the favorite
    print(f"\nPrice-neutral book excess (USD share - Kalshi price), {len(x)} games | corr with home-favorite "
          f"{np.corrcoef(x.excess, (x.sp_close < 0).astype(float))[0, 1]:+.2f} "
          f"(raw USD share {np.corrcoef(x.home_usd / (x.home_usd + x.away_usd), (x.sp_close < 0).astype(float))[0, 1]:+.2f})")
    rec(x, "FADE excess-demand team")
    back = x.assign(fade_home=~x.fade_home, ats=-x.ats, won=~x.won)
    rec(back, "BACK excess-demand team")
    rec(back[fav], "  back it when it's the favorite")
    rec(back[~fav], "  back it when it's the underdog")
    bands(back.assign(a=back.excess.abs()), "a", [0, .05, .1, .2, 1], ["|excess| <5pts", "5-10", "10-20", "20+"],
          "BACK excess-demand team, by size of the excess:")
    dog = x.sp_close > 0
    print(f"  baseline in these games: underdogs covered {100 * np.mean(np.where(dog, mg + x.sp_close > 0, mg + x.sp_close < 0)):.1f}%")

    if args.out:
        df.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
