"""Do public splits and real prediction-market money agree, and who wins when they don't?

  python analysis/money_agreement.py --data-dir store/data [--all-ranked]

Games: ranked vs ranked where the closing spread moved 1+ point toward a side
(the "line moved toward" rule); --all-ranked uses every game with a ranked team.
Signals, all from the backed side's point of view, at kickoff:
  tickets   Action Network % of bets (one source; per-book copies are identical)
  money     Action Network % of money
  kalshi    net Kalshi taker dollars on the backed team, market open -> kickoff
  poly      same on Polymarket
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
from ncaaf_tracker import analyze as an  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402


def splits(data_dir: Path, ids: set) -> pd.DataFrame:
    fl = explosive.load(data_dir, "an_final_lines")
    for c in ("money_pct", "tickets_pct"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce")
    fl = fl[(fl.book == "an_consensus") & (fl.market == "spreads") & ~((fl.money_pct == 0) & (fl.tickets_pct == 0))]
    g = explosive.load(data_dir, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last().reset_index()
    m = Matcher(g)
    rows = []
    for _, e in fl.groupby("event_id"):
        r = e.iloc[0]
        gid, s = m.match(r.home_team, r.away_team, pd.Timestamp(r.commence_time), 12)
        if gid is None or gid not in ids:
            continue
        for _, o in e.iterrows():
            side = "home" if (o.side == "home") == (s[r.home_team] == "home") else "away"
            rows.append((gid, side, o.money_pct, o.tickets_pct))
    return pd.DataFrame(rows, columns=["espn_id", "backed", "money", "tickets"]).drop_duplicates(["espn_id", "backed"])


def rec(d: pd.DataFrame, label: str) -> None:
    w, l_, p = int((d.cover > 0).sum()), int((d.cover < 0).sum()), int((d.cover == 0).sum())
    pct = f"({100 * w / (w + l_):.0f}%)" if w + l_ else ""
    print(f"  {label:58} n={len(d):3d}  ATS {w}-{l_}{'-%d' % p if p else ''} {pct}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--all-ranked", action="store_true")
    args = ap.parse_args(argv)
    D = Path(args.data_dir)
    pd.set_option("display.width", 230)

    df = bt.features(D, bt.games(D))
    rr = df[(df.n_ranked >= 1) if args.all_ranked else (df.n_ranked == 2)].copy()
    mv = rr.sp_close - rr.sp_open
    rr = rr[mv.abs() >= 1].copy()
    home = (rr.sp_close - rr.sp_open) < 0
    rr["backed"] = np.where(home, "home", "away")
    rr["cover"] = np.sign(np.where(home, (rr.hs - rr["as"]) + rr.sp_close, -((rr.hs - rr["as"]) + rr.sp_close)))
    rr = rr.merge(splits(D, set(rr.espn_id)), on=["espn_id", "backed"], how="left")
    # Action Network's late-2025 splits often read exactly 0/100 (or 97-99) on marquee games:
    # not real public splits, so treat money% outside 3-96 as missing.
    bad = ~rr.money.between(3, 96)
    print(f"dropping {int((bad & rr.money.notna()).sum())} implausible money% values (0-2 or 97-100)")
    rr.loc[bad, ["money", "tickets"]] = np.nan

    games = an.build_games(D)
    mt, un = Matcher(games), []
    _, kf = an.build_kalshi(D, games, mt, un)
    _, pf = an.build_poly(D, games, mt, un)
    ko = games.set_index("espn_id").kickoff_dt
    sign = np.where(rr.backed == "home", 1, -1)
    for name, f, col in (("kalshi", kf, "kalshi_flow"), ("poly", pf, "poly_flow")):
        if f is None or f.empty:
            rr[name] = np.nan
            continue
        f = f[f.hour < f.espn_id.map(ko)]
        rr[name] = rr.espn_id.map(f.groupby("espn_id")[col].sum()) * sign
    sgn = lambda v: pd.Series(np.where(v.isna(), np.nan, np.sign(v)), index=v.index)
    rr["t_side"], rr["m_side"] = sgn(rr.tickets - 50), sgn(rr.money - 50)
    rr["k_side"], rr["p_side"] = sgn(rr.kalshi), sgn(rr.poly)

    print(f"games {len(rr)} | AN splits {rr.money.notna().sum()} | Kalshi flow {rr.kalshi.notna().sum()} "
          f"| Polymarket flow {rr.poly.notna().sum()}")
    s = rr.dropna(subset=["money"])
    print("\nAction Network tickets vs money (backed = side the line moved toward):")
    rec(s[(s.t_side > 0) & (s.m_side > 0)], "tickets and money WITH the backed side")
    rec(s[(s.t_side < 0) & (s.m_side < 0)], "tickets and money AGAINST the backed side")
    rec(s[(s.t_side < 0) & (s.m_side > 0)], "disagree: money with, tickets against")
    rec(s[(s.t_side > 0) & (s.m_side < 0)], "disagree: tickets with, money against")
    rec(s[s.m_side < 0], "AN money against the backed side (any tickets)")

    k = rr.dropna(subset=["k_side"])
    print("\nKalshi real money:")
    rec(k[k.k_side > 0], "Kalshi money WITH the backed side")
    rec(k[k.k_side < 0], "Kalshi money AGAINST the backed side")
    km = k.dropna(subset=["m_side"])
    agree = km[km.k_side == km.m_side]
    print(f"  Kalshi side == AN money side: {len(agree)}/{len(km)}")
    rec(km[(km.k_side > 0) & (km.m_side < 0)], "disagree: Kalshi with, AN money against")
    rec(km[(km.k_side < 0) & (km.m_side > 0)], "disagree: Kalshi against, AN money with")
    kp = rr.dropna(subset=["k_side", "p_side"])
    print(f"  Kalshi side == Polymarket side: {(kp.k_side == kp.p_side).sum()}/{len(kp)}")
    rec(kp[kp.k_side != kp.p_side], "Kalshi and Polymarket disagree (backed side's record)")

    cols = ["season", "kickoff", "away", "home", "backed", "tickets", "money", "kalshi", "poly", "cover"]
    out = rr[cols].copy()
    out["kickoff"] = out.kickoff.dt.strftime("%y-%m-%d")
    out[["kalshi", "poly"]] = out[["kalshi", "poly"]].round(0)
    print("\n" + out.sort_values("kickoff").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
