"""What happens between the open and the close: do Kalshi's price, money and order
book move with DraftKings' line, and which one moves first?

  python analysis/intraday.py --data-dir store/data

Per game, an hourly grid from the first DraftKings line to kickoff:
  dk_p       DraftKings no-vig home win chance (Action Network exact line history)
  dk_spread  DraftKings home spread
  k_p        Kalshi home win chance: volume-weighted traded price that hour (carried forward)
  k_flow     Kalshi net taker dollars on home that hour
  k_imb      Kalshi resting-book lean toward home (only hours with an order-book snapshot)
Checks: same-hour co-movement, lead/lag, what Kalshi does around each DraftKings
move, and whether Kalshi-vs-DraftKings gaps close (and toward whom).
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
from ncaaf_tracker import analyze as an  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402


def kalshi_ticker_map(data_dir: Path, matcher: Matcher) -> dict[str, tuple[str, str]]:
    m = explosive.load(data_dir, "kalshi_markets")
    m = m[m.series == "KXNCAAFGAME"]
    out = {}
    for _, d in m.groupby("event_ticker"):
        teams = d.drop_duplicates("ticker")[["ticker", "yes_sub_title"]].dropna()
        if len(teams) != 2:
            continue
        a, b = teams.yes_sub_title.tolist()
        ko = pd.to_datetime(d.kickoff_est, utc=True, errors="coerce").dropna()
        if ko.empty:
            continue
        gid, sides = matcher.match(a, b, ko.iloc[-1], 12)
        if gid is not None:
            for t, name in zip(teams.ticker, teams.yes_sub_title):
                out[t] = (gid, sides[name])
    return out


def dk_history(data_dir: Path, matcher: Matcher) -> pd.DataFrame:
    """DraftKings line changes from Action Network history, mapped to ESPN home/away."""
    h = explosive.load(data_dir, "an_history")
    h = h[h.book == "draftkings"].drop_duplicates(["event_id", "market", "outcome", "updated_at"])
    for c in ("price", "point"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["t"] = pd.to_datetime(h.updated_at, utc=True, format="ISO8601", errors="coerce")
    mp = an.map_events(h.assign(source="an_history"), matcher, [])
    h = h[h.event_id.isin(mp)].copy()
    h["espn_id"] = h.event_id.map(lambda k: mp[k][0])
    aligned = h.event_id.map(lambda k: mp[k][1])
    is_home = (h.outcome == h.home_team) == aligned
    h["eside"] = np.where(h.market == "totals", h.outcome.str.lower(), np.where(is_home, "home", "away"))
    return h


def build_grid(data_dir: Path) -> pd.DataFrame:
    games = an.build_games(data_dir)
    games = games[games["season"].astype(str).str[:4] == "2026"]
    matcher = Matcher(games)
    ko = games.set_index("espn_id").kickoff_dt

    h = dk_history(data_dir, matcher)
    tick = kalshi_ticker_map(data_dir, matcher)
    tr = explosive.load(data_dir, "kalshi_trades").drop_duplicates("trade_id")
    tr = tr[tr.ticker.isin(tick)].copy()
    for c in ("yes_price", "no_price", "count"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    tr["t"] = pd.to_datetime(tr.created_time, utc=True, format="ISO8601", errors="coerce")
    tr["espn_id"] = tr.ticker.map(lambda x: tick[x][0])
    team = tr.ticker.map(lambda x: tick[x][1])
    tr["p_home"] = np.where(team == "home", tr.yes_price, 1 - tr.yes_price)
    paid = np.where(tr.taker_side == "yes", tr.yes_price, tr.no_price) * tr["count"]
    tr["flow"] = np.where((tr.taker_side == "yes") == (team == "home"), paid, -paid)
    tr = tr[tr.t < tr.espn_id.map(ko)]

    bk = explosive.load(data_dir, "kalshi_books")
    bk = bk[bk.ticker.isin(tick)].copy()
    for c in ("price", "size"):
        bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk["t"] = pd.to_datetime(bk.snapshot_ts, utc=True)
    bk["espn_id"] = bk.ticker.map(lambda x: tick[x][0])
    bteam = bk.ticker.map(lambda x: tick[x][1])
    bk["backs"] = np.where(bk.side == "yes", bteam, np.where(bteam == "home", "away", "home"))
    bk["usd"] = bk.price * bk["size"]

    grids = []
    for gid, d in h.groupby("espn_id"):
        k0 = ko.get(gid)
        if k0 is None or pd.isna(k0):
            continue
        d = d[d.t < k0].sort_values("t")
        ml = d[d.market == "h2h"].pivot_table(index="t", columns="eside", values="price", aggfunc="last").ffill()
        if not {"home", "away"} <= set(ml.columns):
            continue
        ml = ml.dropna()
        ph = ml.home.map(bands.implied)
        pa = ml.away.map(bands.implied)
        dkp = (ph / (ph + pa)).rename("dk_p")
        sp = d[(d.market == "spreads") & (d.eside == "home")].set_index("t").point.rename("dk_spread")
        start = max(dkp.index.min(), k0 - pd.Timedelta(days=7)).ceil("h")
        idx = pd.date_range(start, k0.floor("h"), freq="h", inclusive="left")
        if len(idx) < 6:
            continue
        g = pd.DataFrame(index=idx)
        g["dk_p"] = dkp.reindex(dkp.index.union(idx)).ffill().reindex(idx)
        g["dk_spread"] = sp[~sp.index.duplicated(keep="last")].reindex(sp.index.union(idx)).ffill().reindex(idx) if len(sp) else np.nan
        t = tr[tr.espn_id == gid]
        if len(t):
            hr = t.t.dt.ceil("h")
            vw = (t.p_home * t["count"]).groupby(hr).sum() / t["count"].groupby(hr).sum()
            g["k_p"] = vw.reindex(vw.index.union(idx)).ffill().reindex(idx)
            g["k_flow"] = t.flow.groupby(hr).sum().reindex(idx).fillna(0)
            g["k_trades"] = t["count"].groupby(hr).sum().reindex(idx).fillna(0)
        b = bk[bk.espn_id == gid]
        if len(b):
            pv = b.pivot_table(index=b.t.dt.ceil("h"), columns="backs", values="usd", aggfunc="sum")
            if {"home", "away"} <= set(pv.columns):
                g["k_imb"] = ((pv.home - pv.away) / (pv.home + pv.away)).reindex(idx)
        g["espn_id"], g["hours_to_ko"] = gid, (k0 - idx).total_seconds() / 3600
        grids.append(g)
    return pd.concat(grids).rename_axis("hour").reset_index() if grids else pd.DataFrame()


def corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    d = pd.concat([a, b], axis=1).dropna()
    d = d[(d.iloc[:, 0] != 0) | (d.iloc[:, 1] != 0)]
    if len(d) < 20:
        return np.nan, len(d)
    return d.iloc[:, 0].corr(d.iloc[:, 1]), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    args = ap.parse_args(argv)
    g = build_grid(Path(args.data_dir))
    if g.empty:
        print("no overlapping DraftKings history and Kalshi data")
        return 1
    g = g.sort_values(["espn_id", "hour"])
    has_k = g.groupby("espn_id").k_p.transform(lambda s: s.notna().any())
    g = g[has_k]
    grp = g.groupby("espn_id")
    g["d_dk"] = grp.dk_p.diff()
    g["d_k"] = grp.k_p.diff()
    print(f"games with DraftKings history and Kalshi trades: {g.espn_id.nunique()}, game-hours: {len(g)}")

    print("\n1. Same level? Kalshi vs DraftKings win chance")
    both = g.dropna(subset=["dk_p", "k_p"])
    gap = (both.k_p - both.dk_p)
    print(f"   correlation of levels {both.dk_p.corr(both.k_p):.3f} | typical gap {100 * gap.abs().median():.1f} pts "
          f"| gap > 5 pts in {100 * (gap.abs() > 0.05).mean():.0f}% of hours")
    for lo, hi in ((72, 168), (24, 72), (6, 24), (0, 6)):
        x = both[(both.hours_to_ko >= lo) & (both.hours_to_ko < hi)]
        print(f"   {lo:>3}-{hi:<3}h before kickoff: median gap {100 * (x.k_p - x.dk_p).abs().median():.1f} pts (n={len(x)})")

    print("\n2. Same-hour moves: corr(DraftKings change, Kalshi change)")
    r, n = corr(g.d_dk, g.d_k)
    print(f"   price vs price {r:+.3f} (n={n})")
    r, n = corr(g.d_dk, g.k_flow)
    print(f"   DraftKings change vs Kalshi net money {r:+.3f} (n={n})")

    print("\n3. Who moves first? corr(DraftKings change at hour t, Kalshi signal at t+k)")
    print("   k<0: Kalshi moved BEFORE DraftKings | k>0: Kalshi moved AFTER")
    for sig in ("d_k", "k_flow"):
        cells = []
        for k in range(-6, 7):
            r, _ = corr(g.d_dk, grp[sig].shift(-k))
            cells.append(f"{k:+d}:{r:+.2f}")
        print(f"   {sig:7} " + "  ".join(cells))

    print("\n4. Around each DraftKings move of 2+ pts win chance (signed so the move is +)")
    big = g[g.d_dk.abs() >= 0.02].copy()
    sgn = np.sign(big.d_dk)
    for name, col in (("Kalshi price", "k_p"), ("Kalshi net $ (cum.)", "k_flow")):
        cells = []
        for k in (-12, -6, -3, -1, 0, 1, 3, 6, 12):
            if col == "k_p":
                v = (grp.k_p.shift(-k) - grp.k_p.shift(1)).reindex(big.index) * sgn
                cells.append(f"{k:+d}h:{100 * v.mean():+.2f}")
            else:
                cum = grp.k_flow.cumsum()
                v = (cum.shift(-k) - cum.shift(1)).reindex(big.index) * sgn
                cells.append(f"{k:+d}h:${v.median():+,.0f}")
        print(f"   {name:20} " + "  ".join(cells))
    print(f"   ({len(big)} DraftKings moves; Kalshi price change in pts vs the hour before the move)")

    print("\n5. When Kalshi and DraftKings disagree by 3+ pts, who moves toward whom (next 24h)?")
    g["gap"] = g.k_p - g.dk_p
    g["dk_fwd"] = grp.dk_p.shift(-24) - g.dk_p
    g["k_fwd"] = grp.k_p.shift(-24) - g.k_p
    w = g[(g.gap.abs() >= 0.03) & g.dk_fwd.notna() & g.k_fwd.notna()]
    first = w.groupby("espn_id").head(1)                   # first hour each game opens a gap
    toward_k = (np.sign(first.dk_fwd) == np.sign(first.gap)) & (first.dk_fwd != 0)
    toward_dk = (np.sign(first.k_fwd) == -np.sign(first.gap)) & (first.k_fwd != 0)
    print(f"   games with a 3+ pt gap: {len(first)} | DraftKings moved toward Kalshi {toward_k.mean():.0%} | "
          f"Kalshi moved toward DraftKings {toward_dk.mean():.0%}")
    print(f"   average share of the gap closed by DraftKings {100 * (first.dk_fwd / first.gap).clip(-2, 2).mean():.0f}% "
          f"| by Kalshi {100 * (-first.k_fwd / first.gap).clip(-2, 2).mean():.0f}%")

    print("\n6. Kalshi order-book lean vs DraftKings' next 6 hours")
    g["dk_next6"] = grp.dk_p.shift(-6) - g.dk_p
    r, n = corr(g.k_imb, g.dk_next6)
    print(f"   corr(book lean toward home, DraftKings home move next 6h) {r:+.3f} (n={n} snapshots)")
    r, n = corr(g.k_imb, g.d_dk)
    print(f"   corr(book lean, DraftKings move in the same hour) {r:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
