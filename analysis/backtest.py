"""Backtest the rules found on 2026 data against the full 2025 season.

  python analysis/backtest.py --data-dir store/data

Rules are fixed in advance (found on 2026, never tuned on 2025):

Ranked matchups (AP top 25 that week; "both" = ranked vs ranked)
  R1 pick the closing-line favorite                         (baseline)
  R2 pick the team with the better pre-game explosive margin
  R3 pick the team with the better pre-game clutch WPA
  R4 "line against": back the team whose spread moved AGAINST it by 1+ pt
  R5 back the team whose spread moved TOWARD it by 1+ pt
All games
  A1 spread: back the side the line moved toward by 2+ pts (at close)
  A2 moneyline: back the side whose no-vig win chance rose 4+ pts (at close)
  A3 FCS: fade the team better on BOTH explosive margin and clutch WPA (spread, at close)

Every bet is priced at the closing odds (DraftKings / ESPN's provider) and uses
only games played before it within the same season.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import bands  # noqa: E402
import explosive  # noqa: E402


def games(data_dir: Path) -> pd.DataFrame:
    g = explosive.load(data_dir, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last()
    g = g[g["completed"] == "True"].copy()
    rows = []
    for gid, r in g.iterrows():
        try:
            o = json.loads(r["espn_odds_raw"])
        except (TypeError, ValueError):
            o = {}
        ps, ml = o.get("pointSpread") or {}, o.get("moneyline") or {}
        n = lambda d, *p: bands.num(bands._dig(d, *p))
        rows.append({
            "espn_id": gid, "season": int(float(r["season"])), "kickoff": r["kickoff"],
            "division": r["division"], "home": r["home_name"], "away": r["away_name"],
            "home_id": r["home_id"], "away_id": r["away_id"],
            "home_rank": pd.to_numeric(r.get("home_rank"), errors="coerce"),
            "away_rank": pd.to_numeric(r.get("away_rank"), errors="coerce"),
            "hs": float(r["home_score"]), "as": float(r["away_score"]),
            "sp_open": n(ps, "home", "open", "line"), "sp_close": n(ps, "home", "close", "line"),
            "sp_h_odds": n(ps, "home", "close", "odds"), "sp_a_odds": n(ps, "away", "close", "odds"),
            "ml_h_open": n(ml, "home", "open", "odds"), "ml_a_open": n(ml, "away", "open", "odds"),
            "ml_h_close": n(ml, "home", "close", "odds"), "ml_a_close": n(ml, "away", "close", "odds"),
            "provider": r.get("espn_odds_provider"),
        })
    df = pd.DataFrame(rows)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    cur = pd.to_numeric(g["espn_spread_current"], errors="coerce").reindex(df.espn_id)
    df["sp_close"] = df["sp_close"].fillna(pd.Series(cur.to_numpy(), index=df.index))
    df["n_ranked"] = df[["home_rank", "away_rank"]].notna().sum(axis=1)
    return df[df.hs != df["as"]]


def features(data_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    tg = explosive.team_games(data_dir).merge(df[["espn_id", "season"]], on="espn_id")
    tg = tg.sort_values("kickoff")
    grp = tg.groupby(["team", "season"])
    for f in ("expl_margin", "clutch_wpa"):
        tg[f"pre_{f}"] = grp[f].transform(lambda s: s.shift().expanding().mean())
    tg["pre_n"] = grp.cumcount()
    tg = tg[tg.pre_n >= 2]
    h = tg[tg.side == "home"].set_index("espn_id")[["pre_expl_margin", "pre_clutch_wpa"]].add_suffix("_h")
    a = tg[tg.side == "away"].set_index("espn_id")[["pre_expl_margin", "pre_clutch_wpa"]].add_suffix("_a")
    return df.set_index("espn_id").join(h).join(a).reset_index()


# ------------------------------------------------------------------ scoring a pick

def score(df: pd.DataFrame, pick_home: pd.Series) -> dict:
    """pick_home: True = back home, False = back away, NaN = no bet."""
    m = pick_home.notna()
    d, ph = df[m], pick_home[m].astype(bool)
    won = np.where(ph, d.hs > d["as"], d["as"] > d.hs)
    ml_odds = np.where(ph, d.ml_h_close, d.ml_a_close)
    ml_p = [bands.american_profit(o) if w else -1.0 for o, w in zip(ml_odds, won) if not np.isnan(o)]
    res = np.where(ph, (d.hs - d["as"]) + d.sp_close, -((d.hs - d["as"]) + d.sp_close))
    sp_odds = np.where(ph, d.sp_h_odds, d.sp_a_odds)
    ok = ~np.isnan(res)
    sp_p = [0.0 if r == 0 else (bands.american_profit(-110 if np.isnan(o) else o) if r > 0 else -1.0)
            for r, o in zip(res[ok], sp_odds[ok])]
    cov = res[ok][res[ok] != 0] > 0
    return {"bets": int(m.sum()), "win%": 100 * np.mean(won) if len(won) else np.nan,
            "ML ROI%": 100 * np.mean(ml_p) if ml_p else np.nan,
            "cover%": 100 * np.mean(cov) if len(cov) else np.nan,
            "ATS ROI%": 100 * np.mean(sp_p) if sp_p else np.nan,
            "ATS z": (np.mean(sp_p) / (np.std(sp_p, ddof=1) / np.sqrt(len(sp_p)))) if len(sp_p) > 2 else np.nan}


def nan_where(cond: pd.Series, val: pd.Series) -> pd.Series:
    return val.where(cond, np.nan).astype(object).where(cond, np.nan)


def rules(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """name -> (games mask, pick_home series)."""
    p_open = [bands.implied(h) / (bands.implied(h) + bands.implied(a)) if not (np.isnan(h) or np.isnan(a)) else np.nan
              for h, a in zip(df.ml_h_open, df.ml_a_open)]
    p_close = [bands.implied(h) / (bands.implied(h) + bands.implied(a)) if not (np.isnan(h) or np.isnan(a)) else np.nan
               for h, a in zip(df.ml_h_close, df.ml_a_close)]
    df = df.assign(p_open=p_open, p_close=p_close)
    fav_home = pd.Series(np.where(df.p_close.notna(), df.p_close >= 0.5,
                                  np.where(df.sp_close.notna(), df.sp_close <= 0, np.nan)), index=df.index)
    dx = df.pre_expl_margin_h - df.pre_expl_margin_a
    dc = df.pre_clutch_wpa_h - df.pre_clutch_wpa_a
    move_h = df.sp_close - df.sp_open                    # >0: home line got worse (moved against home)
    dp = df.p_close - df.p_open
    out = {}
    for grp, gm in (("both ranked", df.n_ranked == 2), ("any ranked", df.n_ranked >= 1)):
        out[f"{grp} | R1 closing favorite"] = (gm, fav_home)
        out[f"{grp} | R2 better explosive margin"] = (gm, pd.Series(np.where(dx.notna() & (dx != 0), dx > 0, np.nan), index=df.index))
        out[f"{grp} | R3 better clutch WPA"] = (gm, pd.Series(np.where(dc.notna() & (dc != 0), dc > 0, np.nan), index=df.index))
        out[f"{grp} | R4 line moved AGAINST team 1+"] = (gm, pd.Series(np.where(move_h.abs() >= 1, move_h > 0, np.nan), index=df.index))
        out[f"{grp} | R5 line moved TOWARD team 1+"] = (gm, pd.Series(np.where(move_h.abs() >= 1, move_h < 0, np.nan), index=df.index))
    allg = pd.Series(True, index=df.index)
    out["all | A1 spread toward 2+"] = (allg, pd.Series(np.where(move_h.abs() >= 2, move_h < 0, np.nan), index=df.index))
    out["all | A2 ML steamed 4%+"] = (allg, pd.Series(np.where(dp.abs() >= 0.04, dp > 0, np.nan), index=df.index))
    same = np.sign(dx) == np.sign(dc)
    out["FCS | A3 fade better on both"] = (df.division == "FCS", pd.Series(np.where(same & (dx != 0) & dx.notna() & dc.notna(),
                                                                                    dx < 0, np.nan), index=df.index))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    pd.set_option("display.width", 230)

    df = features(data_dir, games(data_dir))
    print("games by season:", df.season.value_counts().sort_index().to_dict())
    for s, d in df.groupby("season"):
        print(f"  {s}: with closing spread {d.sp_close.notna().sum()}, open+close spread "
              f"{(d.sp_open.notna() & d.sp_close.notna()).sum()}, ML close {d.ml_h_close.notna().sum()}, "
              f"ranked-vs-ranked {(d.n_ranked == 2).sum()}, any ranked {(d.n_ranked >= 1).sum()}, "
              f"providers {d.provider.value_counts().head(3).to_dict()}")
    rs = rules(df)
    rows = []
    for name, (gmask, pick) in rs.items():
        for season in sorted(df.season.unique()):
            sm = gmask & (df.season == season)
            r = score(df[sm], pick[sm])
            rows.append({"rule": name, "season": season, **{k: (round(v, 1) if isinstance(v, float) else v)
                                                          for k, v in r.items()}})
    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
