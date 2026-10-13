"""Do explosive and clutch plays predict winners beyond the betting line?

  python analysis/explosive.py --data-dir store/data

From ESPN play-by-play (with ESPN's win probability on every play):
  explosive play  run of 12+ yards or completed pass of 16+ yards
  clutch WPA      win probability a team gained on plays in the 4th quarter / OT
                  while the game was in doubt (home win prob 15-85% before the snap)

Each game uses only the teams' averages from games BEFORE it (no future data).
Baseline is DraftKings' closing line (no-vig moneyline, else the spread).
Questions:
  1. Are the stats stable (does a team's first-half number predict its second half)?
  2. Do they predict the winner better than the closing line alone?
  3. Do they predict covering the closing spread (the part that pays)?
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import bands  # noqa: E402  (closing lines + odds parsing)

RUSH = {"Rush", "Rushing Touchdown"}
PASS_DONE = {"Pass Reception", "Passing Touchdown"}
NOT_SCRIMMAGE = ("Kickoff", "Punt", "Field Goal", "Extra Point", "Timeout", "End ", "Two-point",
                 "Penalty", "Coin", "Safety", "Blocked")


def load(data_dir: Path, table: str, **kw) -> pd.DataFrame:
    files = glob.glob(str(data_dir / table / "*" / "*.csv.gz"))
    return pd.concat((pd.read_csv(f, dtype=str, **kw) for f in files), ignore_index=True) if files else pd.DataFrame()


def team_games(data_dir: Path) -> pd.DataFrame:
    """One row per team per game: explosive plays for/against, clutch WPA, result."""
    p = load(data_dir, "plays").drop_duplicates(["espn_id", "play_id"])
    for c in ("yards", "home_wp_before", "home_wp_after", "period"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    g = load(data_dir, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last()
    g = g[g["completed"] == "True"]
    p = p[p["espn_id"].isin(g.index)]
    p = p.join(g[["home_id", "away_id"]], on="espn_id")

    scrim = ~p["type"].fillna("").str.startswith(NOT_SCRIMMAGE) & p["offense_id"].notna()
    p["explosive"] = scrim & (((p["type"].isin(RUSH)) & (p["yards"] >= 12)) |
                              ((p["type"].isin(PASS_DONE)) & (p["yards"] >= 16)))
    p["scrimmage"] = scrim
    d_home = p["home_wp_after"] - p["home_wp_before"]
    late_close = (p["period"] >= 4) & p["home_wp_before"].between(0.15, 0.85)
    p["home_clutch"] = np.where(late_close, d_home, 0.0)

    rows = []
    for gid, d in p.groupby("espn_id"):
        h, a = g.at[gid, "home_id"], g.at[gid, "away_id"]
        hs, as_ = float(g.at[gid, "home_score"]), float(g.at[gid, "away_score"])
        for team, opp, side, sign, pf, pa in ((h, a, "home", 1, hs, as_), (a, h, "away", -1, as_, hs)):
            off, dfn = d[d["offense_id"] == team], d[d["offense_id"] == opp]
            rows.append({
                "espn_id": gid, "team": team, "opp": opp, "side": side,
                "kickoff": g.at[gid, "kickoff"], "division": g.at[gid, "division"],
                "expl_for": int(off["explosive"].sum()), "expl_against": int(dfn["explosive"].sum()),
                "snaps": int(off["scrimmage"].sum()), "snaps_against": int(dfn["scrimmage"].sum()),
                "clutch_wpa": sign * d["home_clutch"].sum(),
                "won": pf > pa, "margin": pf - pa, "close_game": abs(pf - pa) <= 8,
            })
    tg = pd.DataFrame(rows)
    tg["kickoff"] = pd.to_datetime(tg["kickoff"], utc=True)
    tg["expl_margin"] = tg["expl_for"] - tg["expl_against"]
    tg["expl_rate"] = tg["expl_for"] / tg["snaps"].replace(0, np.nan)
    tg["expl_rate_against"] = tg["expl_against"] / tg["snaps_against"].replace(0, np.nan)
    return tg.sort_values("kickoff")


FEATS = ["expl_margin", "expl_rate", "expl_rate_against", "clutch_wpa"]


def pregame(tg: pd.DataFrame, min_games: int = 2) -> pd.DataFrame:
    """Each team's average of FEATS over its earlier games only."""
    tg = tg.sort_values("kickoff").copy()
    grp = tg.groupby("team")
    for f in FEATS:
        tg[f"pre_{f}"] = grp[f].transform(lambda s: s.shift().expanding().mean())
    tg["pre_n"] = grp.cumcount()
    return tg[tg["pre_n"] >= min_games]


def logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-4) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y) + ridge * w)
    return w


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    pd.set_option("display.width", 220)

    tg = team_games(data_dir)
    print(f"team-games: {len(tg)} ({tg.espn_id.nunique()} games, {tg.team.nunique()} teams)")
    print("\nexplosive plays per team-game:", round(tg.expl_for.mean(), 2),
          "| winners averaged", round(tg[tg.won].expl_margin.mean(), 2), "explosive margin")
    print("won the explosive battle -> won the game:",
          f"{(tg[tg.expl_margin > 0].won.mean()):.1%} (n={int((tg.expl_margin > 0).sum())})")
    print("won the clutch battle (clutch WPA > 0) -> won the game:",
          f"{(tg[tg.clutch_wpa > 0].won.mean()):.1%} (n={int((tg.clutch_wpa > 0).sum())})")

    # 1. stability: first half of each team's games vs second half
    print("\n== 1. Stability (team's first-half average vs second-half average, teams with 4+ games)")
    tg["k"] = tg.groupby("team").cumcount()
    tg["n"] = tg.groupby("team")["team"].transform("size")
    t4 = tg[tg.n >= 4]
    first = t4[t4.k < t4.n // 2].groupby("team")[FEATS].mean()
    second = t4[t4.k >= t4.n // 2].groupby("team")[FEATS].mean()
    for f in FEATS:
        r = first[f].corr(second[f])
        print(f"  {f:18} r = {r:+.2f}  (teams={len(first)})")

    # 2/3. predictive value vs the closing line
    games = bands.load_games(data_dir)[["espn_id", "kickoff", "half", "hs", "as", "sp_h_close", "sp_h_odds",
                                        "sp_a_odds", "ml_h_close", "ml_a_close"]]
    pre = pregame(tg)
    h = pre[pre.side == "home"].set_index("espn_id")
    a = pre[pre.side == "away"].set_index("espn_id")
    both = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    df = games.set_index("espn_id").join(both, how="inner")
    ph = df.apply(lambda r: bands.implied(r.ml_h_close) if not np.isnan(r.ml_h_close) else np.nan, axis=1)
    pa = df.apply(lambda r: bands.implied(r.ml_a_close) if not np.isnan(r.ml_a_close) else np.nan, axis=1)
    spread_p = 0.5 * (1 + (-df["sp_h_close"] / (15.5 * math.sqrt(2))).map(math.erf))
    df["mkt_p"] = (ph / (ph + pa)).fillna(spread_p)
    df = df.dropna(subset=["mkt_p", "sp_h_close"])
    df = df[df.hs != df["as"]]
    df["y"] = (df.hs > df["as"]).astype(float)
    for f in FEATS:
        df[f"d_{f}"] = df[f"pre_{f}_h"] - df[f"pre_{f}_a"]
    print(f"\ngames with both teams' prior data + a closing line: {len(df)}")

    mk = np.log(df.mkt_p / (1 - df.mkt_p)).clip(-8, 8)
    tr, te = df.half == "early", df.half == "late"
    print("\n== 2. Predicting the winner (fit on early half, scored on late half; lower log loss = better)")
    base = logloss(df.mkt_p[te], df.y[te])
    print(f"  closing line alone:            log loss {base:.4f}  (n={int(te.sum())})")
    for feats in (["d_expl_margin"], ["d_clutch_wpa"], ["d_expl_rate", "d_expl_rate_against"],
                  ["d_expl_margin", "d_clutch_wpa"]):
        Z = df[feats].fillna(0).to_numpy()
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        X = np.column_stack([np.ones(len(df)), mk, (Z - mu) / sd])
        w = logit_fit(X[tr.to_numpy()], df.y[tr].to_numpy())
        p = 1 / (1 + np.exp(-X[te.to_numpy()] @ w))
        print(f"  line + {' + '.join(feats):38} log loss {logloss(p, df.y[te].to_numpy()):.4f}  "
              f"weights {np.round(w[2:], 3)}")

    print("\n== 3. Against the closing spread: back the side with the better pre-game number (bet at close)")
    margin = df.hs - df["as"]
    res_h = margin + df.sp_h_close
    for f in ("d_expl_margin", "d_expl_rate", "d_clutch_wpa"):
        for thresh_q in (0.0, 0.5):
            cut = df[f].abs().quantile(thresh_q)
            pick = df[df[f].abs() > cut]
            side_home = pick[f] > 0
            res = np.where(side_home, res_h[pick.index], -res_h[pick.index])
            odds = np.where(side_home, pick.sp_h_odds.fillna(-110), pick.sp_a_odds.fillna(-110))
            prof = np.where(res == 0, 0.0, np.where(res > 0, [bands.american_profit(o) for o in odds], -1.0))
            s = pd.Series(prof, index=pick.index)
            lab = "all games" if thresh_q == 0 else "top half by gap"
            e, l_ = s[pick.half == "early"], s[pick.half == "late"]
            print(f"  {f:16} {lab:16} bets {len(s):4d}  W {int((res > 0).sum()):4d}  ROI {100 * s.mean():+5.1f}%  "
                  f"(early {100 * e.mean():+5.1f}%, late {100 * l_.mean():+5.1f}%)")

    print("\n== 4. FADE: back the side with the WORSE pre-game number (bet at its own closing odds)")

    def ats(mask_home: pd.Series, sub: pd.DataFrame) -> pd.Series:
        res = np.where(mask_home, res_h[sub.index], -res_h[sub.index])
        odds = np.where(mask_home, sub.sp_h_odds.fillna(-110), sub.sp_a_odds.fillna(-110))
        return pd.Series(np.where(res == 0, 0.0, np.where(res > 0, [bands.american_profit(o) for o in odds], -1.0)),
                         index=sub.index), res

    def show(label, sub, mask_home):
        s, res = ats(mask_home, sub)
        n, w = len(s), int((res > 0).sum())
        lo = int((res < 0).sum())
        se = s.std(ddof=1) / np.sqrt(n)
        by = " ".join(f"{k}:{100 * s[sub.half == k].mean():+.1f}%({int((sub.half == k).sum())})" for k in ("early", "late"))
        dv = " ".join(f"{k}:{100 * s[sub.division_h == k].mean():+.1f}%({int((sub.division_h == k).sum())})"
                      for k in ("FBS", "FCS"))
        print(f"  {label:44} {w}-{lo}  ROI {100 * s.mean():+5.1f}%  z {s.mean() / se:+.2f} | {by} | {dv}")

    for f in ("d_expl_margin", "d_expl_rate", "d_clutch_wpa"):
        sub = df[df[f] != 0]
        show(f"fade better {f[2:]}", sub, sub[f] < 0)
    both = df[(np.sign(df.d_expl_margin) == np.sign(df.d_clutch_wpa)) & (df.d_expl_margin != 0)]
    show("fade team better on BOTH explosive + clutch", both, both.d_expl_margin < 0)
    mixed = df[(np.sign(df.d_expl_margin) != np.sign(df.d_clutch_wpa)) & (df.d_expl_margin != 0) & (df.d_clutch_wpa != 0)]
    show("explosive+, clutch-: back explosive side", mixed, mixed.d_expl_margin > 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
