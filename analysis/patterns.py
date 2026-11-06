"""Seven pattern searches over everything collected, each with its own guard rail.

  python analysis/patterns.py --data-dir store/data [--only 1,2,3] [--out reports/patterns.md]

1 kalshi_lag   After DraftKings' no-vig win chance jumps 4+ pts in an hour, buy that team on Kalshi
               the NEXT hour (real traded VWAP, fee + 1c slippage) and exit at kickoff / settle.
               Guard: same entry at random hours with no DraftKings move.            (2026)
2 stale_lines  Soft-book spreads that beat Pinnacle's no-vig price by 2%+ EV at the same
               snapshot. Graded at the book's own line; CLV vs Pinnacle's close.     (2026)
3 clv          Closing line value of each signal at the moment it first fires (hourly
               snapshots): did the line keep moving our way? Bootstrap 95% CI.       (2026)
4 timing       Early vs late line moves, and 1.5+ pt "steam" moves inside one hour.  (2026)
5 regime       The least-money + line-move method split by week, slot, size, fav/dog,
               home/away, ranked, FBS/FCS, 2025 vs 2026 side by side.
6 football     Play-by-play success rate, turnover margin and margin "luck" (points vs
               success), season-to-date before each game, vs the closing spread.
7 model        Logistic regression + gradient boosting on all side features, trained on one
               season and tested on the other; compared with a coin flip and the market.
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
import bands  # noqa: E402
import edge_search as es  # noqa: E402
import explosive  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402

OUT: list[str] = []
RNG = np.random.default_rng(11)


def say(s: str = "") -> None:
    print(s, flush=True)
    OUT.append(s)


def profit(res, odds) -> np.ndarray:
    return np.array([0.0 if r == 0 else (bands.american_profit(-110 if pd.isna(o) else o) if r > 0 else -1.0)
                     for r, o in zip(res, odds)])


def rec(res, odds=None) -> str:
    res = np.asarray(res, float)
    res = res[~np.isnan(res)]
    w, l_ = int((res > 0).sum()), int((res < 0).sum())
    p = profit(res, [np.nan] * len(res) if odds is None else odds)
    return f"{w}-{l_} ({100 * w / max(w + l_, 1):.1f}%) ROI {100 * p.mean() if len(p) else 0:+.1f}% n={len(res)}"


def ci(x: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    b = RNG.choice(x, (n, len(x))).mean(axis=1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def espn_games(D: Path) -> pd.DataFrame:
    g = explosive.load(D, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last().reset_index()
    g["kickoff_dt"] = pd.to_datetime(g.kickoff, utc=True)
    return g


def resolve(df: pd.DataFrame, m: Matcher, when_col: str) -> pd.DataFrame:
    """Map rows with home_team/away_team/when to ESPN ids, and whether the source's home is ESPN's away."""
    keys = df[["home_team", "away_team", when_col]].drop_duplicates(["home_team", "away_team"])
    ids, flip = {}, {}
    for _, k in keys.iterrows():
        gid, sides = m.match(k.home_team, k.away_team, k[when_col], 12)
        if gid is not None:
            ids[(k.home_team, k.away_team)] = gid
            flip[(k.home_team, k.away_team)] = sides[k.home_team] != "home"
    key = list(zip(df.home_team, df.away_team))
    df = df.assign(espn_id=[ids.get(x) for x in key], flipped=[flip.get(x, False) for x in key])
    return df[df.espn_id.notna()]


def espn_side(df: pd.DataFrame) -> pd.Series:
    is_src_home = df.outcome == df.home_team
    return pd.Series(np.where(is_src_home != df.flipped, "home", "away"), index=df.index)


def results(D: Path) -> pd.DataFrame:
    g = bt.games(D)
    return g.set_index("espn_id")


# ------------------------------------------------------------------ 1 kalshi lag

def kalshi_lag(D: Path, R: pd.DataFrame) -> None:
    import intraday
    say("\n## 1. Kalshi lag: buy on Kalshi after a DraftKings move (2026)")
    g = intraday.build_grid(D)
    if g.empty:
        say("no grid")
        return
    g = g.sort_values(["espn_id", "hour"])
    g["d_dk"] = g.groupby("espn_id").dk_p.diff()
    g["k_next"] = g.groupby("espn_id").k_p.shift(-1)          # next hour's traded VWAP (entry)
    g["tr_next"] = g.groupby("espn_id").k_trades.shift(-1)
    last = g.groupby("espn_id").k_p.last()                      # last traded price before kickoff
    g["k_close"] = g.espn_id.map(last)
    won_home = (R.hs > R["as"]).reindex(g.espn_id.unique())
    g["home_won"] = g.espn_id.map(won_home)

    def trades(mask, toward_home):
        d = g[mask & (g.tr_next > 0) & g.k_next.notna()].copy()
        d = d.groupby("espn_id").head(1)                        # first qualifying hour per game
        th = toward_home.reindex(d.index)
        p_in = np.where(th, d.k_next, 1 - d.k_next)             # price of the side we buy
        p_out = np.where(th, d.k_close, 1 - d.k_close)
        won = np.where(th, d.home_won, ~d.home_won.astype(bool))
        fee = 0.07 * p_in * (1 - p_in) + 0.01                   # Kalshi taker fee + 1c slippage
        to_close = (p_out - p_in - fee) / p_in                  # return on stake, exit at kickoff price
        settle = (np.where(won.astype(bool), 1.0, 0.0) - p_in - fee) / p_in
        return d, to_close, settle

    for thr in (0.04, 0.06):
        mv = g.d_dk.abs() >= thr
        d, tc, st = trades(mv, g.d_dk > 0)
        m1, lo1, hi1 = ci(tc)
        m2, lo2, hi2 = ci(st)
        say(f"- DraftKings win chance moved {thr * 100:.0f}+ pts in an hour: n={len(d)} games | "
            f"exit at kickoff {m1 * 100:+.1f}% per bet (95% CI {lo1 * 100:+.1f} to {hi1 * 100:+.1f}) | "
            f"hold to result {m2 * 100:+.1f}% ({lo2 * 100:+.1f} to {hi2 * 100:+.1f})")
    # guard: random hours with no DraftKings move, random side
    quiet = g.d_dk.abs() < 0.005
    pick = pd.Series(RNG.random(len(g)) < 0.5, index=g.index)
    d, tc, st = trades(quiet & (RNG.random(len(g)) < 0.15), pick)
    m1, lo1, hi1 = ci(tc)
    say(f"- Guard, random quiet hours & random side: n={len(d)} | exit at kickoff {m1 * 100:+.1f}% "
        f"({lo1 * 100:+.1f} to {hi1 * 100:+.1f})")
    # gap version: Kalshi behind DraftKings by 4+ pts
    gap = (g.dk_p - g.k_p)
    d, tc, st = trades(gap.abs() >= 0.04, gap > 0)
    m1, lo1, hi1 = ci(tc)
    say(f"- Kalshi 4+ pts behind DraftKings, buy toward DraftKings: n={len(d)} | exit at kickoff "
        f"{m1 * 100:+.1f}% ({lo1 * 100:+.1f} to {hi1 * 100:+.1f})")


# ------------------------------------------------------------------ 2 stale lines

SOFT = ["draftkings", "fanduel", "betmgm", "caesars", "bet365", "betrivers"]


def snapshots(D: Path, m: Matcher, table: str, books: list[str]) -> pd.DataFrame:
    o = explosive.load(D, table)
    o = o[(o.market == "spreads") & o.book.isin(books)].copy()
    for c in ("price", "point"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = resolve(o, m, "commence_time")
    o["side"] = espn_side(o)
    o["t"] = pd.to_datetime(o.snapshot_ts, utc=True)
    o["ko"] = pd.to_datetime(o.commence_time, utc=True)
    o = o[o.t < o.ko]
    return o[["snapshot_ts", "t", "ko", "espn_id", "book", "side", "point", "price"]].dropna(subset=["point"])


def stale_lines(D: Path, R: pd.DataFrame, m: Matcher) -> None:
    say("\n## 2. Stale soft-book lines vs Pinnacle (2026)")
    a = snapshots(D, m, "an_odds", SOFT)
    p = snapshots(D, m, "pinnacle_odds", ["pinnacle"])
    pv = p.pivot_table(index=["snapshot_ts", "espn_id", "side"], values=["point", "price"], aggfunc="last")
    pv = pv.reset_index()
    other = pv.assign(side=np.where(pv.side == "home", "away", "home"))[["snapshot_ts", "espn_id", "side", "price"]]
    pv = pv.merge(other.rename(columns={"price": "price_other"}), on=["snapshot_ts", "espn_id", "side"])
    imp = lambda x: np.array([bands.implied(v) if not pd.isna(v) else np.nan for v in x])
    pv["p_fair"] = imp(pv.price) / (imp(pv.price) + imp(pv.price_other))
    pv = pv.rename(columns={"point": "pin_point", "price": "pin_price"})
    j = a.merge(pv, on=["snapshot_ts", "espn_id", "side"])
    j["d_pts"] = j.point - j.pin_point                           # extra points vs Pinnacle (+ = better)
    j["p"] = (j.p_fair + 0.03 * j.d_pts).clip(0.02, 0.98)       # ~3% cover chance per half-point-ish
    dec = np.array([1 + bands.american_profit(v) if not pd.isna(v) else np.nan for v in j.price])
    j["ev"] = j.p * (dec - 1) - (1 - j.p)
    say(f"- book/Pinnacle side-snapshots compared: {len(j):,} ({j.espn_id.nunique()} games)")
    close = p.sort_values("t").groupby(["espn_id", "side"]).point.last()
    for thr in (0.02, 0.04):
        f = j[(j.ev >= thr)].sort_values("t").groupby(["espn_id", "book", "side"]).head(1)
        f = f[f.espn_id.isin(R.index)]
        if f.empty:
            say(f"- EV >= {thr * 100:.0f}%: none")
            continue
        sgn = np.where(f.side == "home", 1, -1)
        margin = sgn * (f.espn_id.map(R.hs) - f.espn_id.map(R["as"]))
        res = margin + f.point
        clv = f.point - [close.get((e, s), np.nan) for e, s in zip(f.espn_id, f.side)]
        mc, lo, hi = ci(clv)
        say(f"- EV >= {thr * 100:.0f}% vs Pinnacle: {rec(res, f.price)} | avg {f.d_pts.mean():+.2f} pts vs Pinnacle | "
            f"CLV vs Pinnacle close {mc:+.2f} pts ({lo:+.2f} to {hi:+.2f}) | books: "
            + ", ".join(f"{b} {n}" for b, n in f.book.value_counts().items()))
    # one bet per game-side (best EV book), and the same side taken at Pinnacle's number instead
    f = j[(j.ev >= 0.02) & j.espn_id.isin(R.index)].sort_values("t").groupby(["espn_id", "side"]).head(5)
    f = f.sort_values("ev").groupby(["espn_id", "side"]).tail(1)
    sgn = np.where(f.side == "home", 1, -1)
    margin = sgn * (f.espn_id.map(R.hs) - f.espn_id.map(R["as"]))
    say(f"- one bet per game-side (best book, EV >= 2%): {rec(margin + f.point, f.price)} | same side at "
        f"Pinnacle's number at that time: {rec(margin + f.pin_point, f.pin_price)}")
    for b, d in f.groupby("book"):
        sg = np.where(d.side == "home", 1, -1)
        mg = sg * (d.espn_id.map(R.hs) - d.espn_id.map(R["as"]))
        say(f"  - {b}: {rec(mg + d.point, d.price)}")
    base = j[j.espn_id.isin(R.index)].sort_values("t").groupby(["espn_id", "book", "side"]).head(1)
    sgn = np.where(base.side == "home", 1, -1)
    res = sgn * (base.espn_id.map(R.hs) - base.espn_id.map(R["as"])) + base.point
    say(f"- Guard, every first soft-book line (both sides): {rec(res, base.price)}")


# ------------------------------------------------------------------ 3 clv

def clv(D: Path, R: pd.DataFrame, m: Matcher) -> None:
    say("\n## 3. Closing line value of each signal when it first fires (2026 hourly snapshots)")
    a = snapshots(D, m, "an_odds", ["draftkings", "an_open"])
    dk = a[a.book == "draftkings"]
    op = a[a.book == "an_open"].sort_values("t").groupby(["espn_id", "side"]).point.last()
    sp = explosive.load(D, "an_splits")
    sp = sp[sp.market == "spreads"].copy()
    sp = resolve(sp, m, "commence_time")
    sp["eside"] = np.where((sp.side == "home") != sp.flipped, "home", "away")
    for c in ("tickets_pct", "money_pct"):
        sp[c] = pd.to_numeric(sp[c], errors="coerce")
    sp = sp[sp.money_pct.between(3, 96)][["snapshot_ts", "espn_id", "eside", "tickets_pct", "money_pct"]]
    x = dk.merge(sp.rename(columns={"eside": "side"}), on=["snapshot_ts", "espn_id", "side"])
    x["open"] = [op.get((e, s), np.nan) for e, s in zip(x.espn_id, x.side)]
    x["toward"] = x.open - x.point                               # + = line moved toward this side
    close = dk.sort_values("t").groupby(["espn_id", "side"]).point.last()
    x["close"] = [close.get((e, s), np.nan) for e, s in zip(x.espn_id, x.side)]
    x["clv"] = x.point - x.close                                 # + = we got a better number than the close
    sgn = np.where(x.side == "home", 1, -1)
    x["res"] = np.where(x.espn_id.isin(R.index),
                        sgn * (x.espn_id.map(R.hs) - x.espn_id.map(R["as"])) + x.point, np.nan)
    x = x.sort_values("t")
    sigs = {
        "least money + moved toward (move>=1, money<50%)": (x.toward >= 1) & (x.money_pct < 50),
        "strong version (move>=2, money<30%)": (x.toward >= 2) & (x.money_pct < 30),
        "sharp split (tickets>=70% other side, money majority here)": (100 - x.tickets_pct >= 70) & (x.money_pct > 50),
        "fade 90% tickets": (100 - x.tickets_pct >= 90),
        "guard: every side at its first snapshot": x.groupby(["espn_id", "side"]).cumcount() == 0,
        "guard: side with MOST money, moved toward (move>=1, money>50%)": (x.toward >= 1) & (x.money_pct > 50),
    }
    for name, msk in sigs.items():
        f = x[msk].groupby(["espn_id", "side"]).head(1)
        mc, lo, hi = ci(f.clv)
        say(f"- {name}: n={len(f)} | CLV {mc:+.2f} pts (95% CI {lo:+.2f} to {hi:+.2f}), "
            f"beat close {100 * (f.clv > 0).mean():.0f}% / matched {100 * (f.clv == 0).mean():.0f}% | ATS {rec(f.res)}")


# ------------------------------------------------------------------ 4 timing

def timing(D: Path, R: pd.DataFrame, m: Matcher) -> None:
    import intraday
    say("\n## 4. When the line moved: early vs late, and steam (2026, exact DraftKings history)")
    h = intraday.dk_history(D, m)
    h = h[(h.market == "spreads") & (h.eside == "home")].dropna(subset=["point", "t"])
    ko = espn_games(D).set_index("espn_id").kickoff_dt
    h = h[h.t < h.espn_id.map(ko)].sort_values("t")
    sides = es.splits(D)
    mon = sides.set_index(["espn_id", "side"]).sp_mon
    rows, steam = [], []
    for gid, d in h.groupby("espn_id"):
        if gid not in R.index:
            continue
        k0 = ko[gid]
        first, last = d.point.iloc[0], d.point.iloc[-1]
        before = d[d.t <= k0 - pd.Timedelta(hours=24)]
        at24 = before.point.iloc[-1] if len(before) else first
        total, late = last - first, last - at24
        rows.append((gid, total, late, last))
        s = d.set_index("t").point
        for t0, v0 in s.items():                                # 1.5+ pt move within 60 minutes
            w = s[(s.index > t0) & (s.index <= t0 + pd.Timedelta(minutes=60))]
            if len(w) and abs(w.iloc[-1] - v0) >= 1.5:
                steam.append((gid, np.sign(-(w.iloc[-1] - v0)), (k0 - t0).total_seconds() / 3600))
                break
    t = pd.DataFrame(rows, columns=["espn_id", "total", "late", "close"])
    t["toward_home"] = t.total < 0
    margin = t.espn_id.map(R.hs) - t.espn_id.map(R["as"])
    t["res_home"] = margin + t.close
    t = t[t.total.abs() >= 1]
    side = np.where(t.toward_home, "home", "away")
    t["res"] = np.where(t.toward_home, t.res_home, -t.res_home)
    t["mon"] = [mon.get((e, s), np.nan) for e, s in zip(t.espn_id, side)]
    t["late_share"] = t.late / t.total
    say(f"- games with a 1+ pt DraftKings move: {len(t)}")
    for lab, msk in (("most of the move in the final 24h", t.late_share >= 0.5),
                     ("most of the move earlier", t.late_share < 0.5)):
        d = t[msk]
        say(f"  - {lab}: side moved toward {rec(d.res)} | with money <50% on it: {rec(d[d.mon < 50].res)}")
    s = pd.DataFrame(steam, columns=["espn_id", "toward_home", "hrs_before"])
    if len(s):
        cl = t.set_index("espn_id")
        s = s[s.espn_id.isin(R.index)]
        mg = s.espn_id.map(R.hs) - s.espn_id.map(R["as"])
        cls = s.espn_id.map(R.sp_close)
        res = np.where(s.toward_home > 0, mg + cls, -(mg + cls))
        say(f"- steam (1.5+ pts inside an hour), back the side it moved toward at the close: {rec(res)}")
        for lab, msk in (("steam in final 24h", s.hrs_before <= 24), ("steam earlier", s.hrs_before > 24)):
            say(f"  - {lab}: {rec(res[msk.values])}")
        del cl


# ------------------------------------------------------------------ 5 regime

def side_table(D: Path) -> pd.DataFrame:
    g = bt.games(D)
    g = g[g.sp_close.notna() & g.sp_open.notna()].copy()
    sp = es.splits(D)
    start = g.groupby("season").kickoff.transform("min")
    g["week"] = ((g.kickoff - start).dt.days // 7 + 1).astype(int)
    et = g.kickoff.dt.tz_convert("America/New_York")
    g["slot"] = np.select([et.dt.dayofweek != 5, et.dt.hour < 15, et.dt.hour < 19], ["weekday", "Sat early", "Sat afternoon"], "Sat night")
    rows = []
    for side, sg in (("home", 1), ("away", -1)):
        x = g.merge(sp[sp.side == side], on="espn_id")
        line = x.sp_close if side == "home" else -x.sp_close
        own = (x.home_rank if side == "home" else x.away_rank).notna()
        opp = (x.away_rank if side == "home" else x.home_rank).notna()
        rows.append(pd.DataFrame({
            "espn_id": x.espn_id, "season": x.season, "week": x.week, "slot": x.slot, "division": x.division,
            "side": side, "line": line, "res": sg * (x.hs - x["as"]) + line,
            "odds": x.sp_h_odds if side == "home" else x.sp_a_odds,
            "won": sg * (x.hs - x["as"]) > 0, "ml": x.ml_h_close if side == "home" else x.ml_a_close,
            "move": -sg * (x.sp_close - x.sp_open), "mon": x.sp_mon, "tix": x.sp_tix,
            "ml_mon": x.ml_mon, "ml_tix": x.ml_tix, "own_rk": own, "opp_rk": opp}))
    s = pd.concat(rows, ignore_index=True)
    return s[s.mon.between(3, 96) & s.res.notna()]


def regime(S: pd.DataFrame) -> None:
    say("\n## 5. Where does the 2026 edge come from? (method: move>=1 toward side, money<50%)")
    m = S[(S.move >= 1) & (S.mon < 50)].copy()
    m["size"] = pd.cut(m.line.abs(), [-1, 3.25, 7.25, 14.25, 99], labels=["0-3", "3.5-7", "7.5-14", "14.5+"]).astype(str)
    m["fav"] = np.where(m.line < 0, "favorite", "underdog")
    m["wk"] = pd.cut(m.week, [0, 4, 8, 11, 30], labels=["wk 1-4", "wk 5-8", "wk 9-11", "wk 12+"]).astype(str)
    m["rank"] = np.select([m.own_rk & m.opp_rk, m.own_rk | m.opp_rk], ["both ranked", "one ranked"], "unranked")
    m["mvsz"] = pd.cut(m.move, [0.9, 1.75, 2.75, 99], labels=["move 1-1.5", "move 2-2.5", "move 3+"]).astype(str)
    say("| split | bucket | 2025 | 2026 |")
    say("|---|---|---|---|")
    for col in ("wk", "slot", "division", "fav", "side", "size", "rank", "mvsz"):
        for b, d in m.groupby(col):
            a, c = d[d.season == 2025], d[d.season == 2026]
            say(f"| {col} | {b} | {rec(a.res, a.odds)} | {rec(c.res, c.odds)} |")
    say("- read: a bucket matters only if it is strong in BOTH seasons; a bucket strong only in 2026 is a "
        "2026 effect, not a lasting one.")


# ------------------------------------------------------------------ 6 football

def team_stats(D: Path, slope: float | None = None) -> pd.DataFrame:
    """Per team-game success rate, turnovers and luck. slope=None fits points-per-success-rate on
    the data; forward rules pass the frozen value so the definition never drifts."""
    p = explosive.load(D, "plays").drop_duplicates(["espn_id", "play_id"])
    for c in ("down", "distance", "yards", "period"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    scrim = ~p.type.fillna("").str.startswith(explosive.NOT_SCRIMMAGE) & p.offense_id.notna() & p.down.between(1, 4)
    p = p[scrim]
    need = np.select([p.down == 1, p.down == 2], [0.5 * p.distance, 0.7 * p.distance], p.distance)
    p["success"] = p.yards >= need
    off = p.groupby(["espn_id", "offense_id"]).agg(sr=("success", "mean"), plays=("success", "size"),
                                                  ypp=("yards", "mean")).reset_index()
    b = explosive.load(D, "box_team")
    b = b[b.stat == "turnovers"]
    b["to"] = pd.to_numeric(b.value, errors="coerce")
    g = espn_games(D)
    g = g[g.completed == "True"]
    rows = []
    for _, r in g.iterrows():
        for team, opp, side, pf, pa in ((r.home_id, r.away_id, "home", r.home_score, r.away_score),
                                        (r.away_id, r.home_id, "away", r.away_score, r.home_score)):
            rows.append((r.espn_id, str(team), str(opp), side, r.kickoff_dt, float(pf), float(pa)))
    t = pd.DataFrame(rows, columns=["espn_id", "team", "opp", "side", "kickoff", "pf", "pa"])
    o = off.rename(columns={"offense_id": "team"})
    t = t.merge(o, on=["espn_id", "team"], how="left")
    t = t.merge(o.rename(columns={"team": "opp", "sr": "sr_against", "plays": "plays_against", "ypp": "ypp_against"}),
                on=["espn_id", "opp"], how="left")
    tov = b.set_index(["espn_id", "team_id"]).to
    t["to_lost"] = [tov.get((e, x), np.nan) for e, x in zip(t.espn_id, t.team)]
    t["to_won"] = [tov.get((e, x), np.nan) for e, x in zip(t.espn_id, t.opp)]
    t = t.dropna(subset=["sr", "sr_against"])
    t["margin"] = t.pf - t.pa
    t["sr_diff"] = t.sr - t.sr_against
    t["to_margin"] = t.to_won - t.to_lost
    fit = np.polyfit(t.sr_diff, t.margin, 1) if slope is None else (slope, 0.0)  # margin from success rate
    t["luck"] = t.margin - np.polyval(fit, t.sr_diff)          # points beyond what success rate says
    t = t.sort_values("kickoff")
    t["season"] = np.where(t.kickoff.dt.month >= 7, t.kickoff.dt.year, t.kickoff.dt.year - 1)
    grp = t.groupby(["team", "season"])
    for f in ("sr_diff", "to_margin", "luck"):
        t[f"pre_{f}"] = grp[f].transform(lambda s: s.shift().expanding().mean())
    t["pre_n"] = grp.cumcount()
    return t


def football(D: Path, S: pd.DataFrame) -> pd.DataFrame:
    say("\n## 6. Football stats vs the spread (season-to-date before each game, 3+ prior games)")
    t = team_stats(D)
    pre = t[t.pre_n >= 3][["espn_id", "side", "pre_sr_diff", "pre_to_margin", "pre_luck"]]
    x = S.merge(pre, on=["espn_id", "side"])
    opp = pre.assign(side=np.where(pre.side == "home", "away", "home")).rename(
        columns={"pre_sr_diff": "o_sr", "pre_to_margin": "o_to", "pre_luck": "o_luck"})
    x = x.merge(opp, on=["espn_id", "side"])
    x["d_sr"], x["d_to"], x["d_luck"] = x.pre_sr_diff - x.o_sr, x.pre_to_margin - x.o_to, x.pre_luck - x.o_luck
    tests = {
        "underdog with the better success-rate differential (by 5+ pts)": (x.line > 0) & (x.d_sr >= 0.05),
        "favorite with the WORSE success-rate differential (fade it = back the other side)": (x.line > 0) & (x.d_sr >= 0) & (x.d_sr < 0.05),
        "fade the team +1/game better in turnover margin (back its opponent)": x.d_to <= -1,
        "fade the 'lucky' team (+7 pts/game more luck) (back its opponent)": x.d_luck <= -7,
        "back the 'unlucky' team, 10+ pts/game less luck than its opponent": x.d_luck <= -10,
    }
    for name, msk in tests.items():
        d = x[msk]
        say(f"- {name}: 2025 {rec(d[d.season == 2025].res, d[d.season == 2025].odds)} | "
            f"2026 {rec(d[d.season == 2026].res, d[d.season == 2026].odds)}")
    return x[["espn_id", "side", "d_sr", "d_to", "d_luck"]]


# ------------------------------------------------------------------ 7 model

def model(S: pd.DataFrame, F: pd.DataFrame) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    say("\n## 7. Model on everything: train on one season, test on the other")
    x = S.merge(F, on=["espn_id", "side"], how="left")
    x = x[x.res != 0].copy()
    x["y"] = (x.res > 0).astype(int)
    x["home"] = (x.side == "home").astype(int)
    x["gap"] = x.mon - x.tix
    x["ml_gap"] = x.ml_mon - x.ml_tix
    x["fcs"] = (x.division == "FCS").astype(int)
    feats = ["move", "mon", "tix", "gap", "ml_mon", "ml_tix", "ml_gap", "line", "home", "own_rk", "opp_rk",
             "fcs", "d_sr", "d_to", "d_luck"]
    x[feats] = x[feats].astype(float)
    x = x[x.side == "home"]                                     # one row per game (away = mirror)
    for train, test in ((2025, 2026), (2026, 2025)):
        a, b = x[x.season == train], x[x.season == test]
        for name, mdl in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))),
                          ("boosting", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                       max_iter=200, l2_regularization=1.0))):
            A = a[feats].fillna(a[feats].median())
            B = b[feats].fillna(a[feats].median())
            mdl.fit(A, a.y)
            p = mdl.predict_proba(B)[:, 1]
            ll, ll0 = log_loss(b.y, p), log_loss(b.y, np.full(len(b), 0.5))
            auc = roc_auc_score(b.y, p)
            line = [f"- {name}, train {train} -> test {test}: log-loss {ll:.4f} vs coin flip {ll0:.4f}, AUC {auc:.3f}"]
            for thr in (0.53, 0.55, 0.58):
                pick_home = p >= thr
                pick_away = p <= 1 - thr
                res = np.r_[b.res.values[pick_home], -b.res.values[pick_away]]
                line.append(f"p>={thr}: {rec(res)}")
            say(" | ".join(line))
    say("- read: AUC 0.50 = no skill; log-loss must beat the coin flip; picks must clear 52.4% in BOTH directions.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--only", default="1,2,3,4,5,6,7")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    D = Path(a.data_dir)
    only = {int(x) for x in a.only.split(",")}
    pd.set_option("display.width", 220)
    say(f"# Pattern search ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d})")
    R = results(D)
    m = Matcher(espn_games(D))
    if 1 in only:
        kalshi_lag(D, R)
    if 2 in only:
        stale_lines(D, R, m)
    if 3 in only:
        clv(D, R, m)
    if 4 in only:
        timing(D, R, m)
    S = side_table(D) if only & {5, 6, 7} else None
    if 5 in only:
        regime(S)
    F = football(D, S) if only & {6, 7} else None
    if 7 in only:
        model(S, F)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(OUT) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
