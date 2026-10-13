"""Band line moves and closing prices, then check wins/losses and ROI.

  python analysis/bands.py --data-dir store/data --out out/bands

Source: ESPN's DraftKings open/close (line + actual odds) for every finished game.
Every bet is priced at the CLOSING odds, so a band is something you could
actually bet: the open->close move is known at kickoff.

To separate signal from luck, games are split by kickoff date into an early
half (find) and a late half (check). A real edge should be positive in both.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def american_profit(odds: float) -> float:
    """Profit per 1 unit staked on a win."""
    return odds / 100 if odds > 0 else 100 / -odds


def implied(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def num(v):
    try:
        s = str(v).strip().lower().lstrip("ou")
        return {"even": 100.0, "pk": 0.0}.get(s, float(s))
    except (TypeError, ValueError):
        return np.nan


def load_games(data_dir: Path) -> pd.DataFrame:
    g = pd.concat((pd.read_csv(f, dtype=str) for f in glob.glob(str(data_dir / "espn_games/*/*.csv.gz"))),
                  ignore_index=True).sort_values("snapshot_ts")
    last = g.groupby("espn_id").last()            # last non-null of every column
    fin = last[last["completed"] == "True"].copy()
    rows = []
    for gid, r in fin.iterrows():
        try:
            o = json.loads(r["espn_odds_raw"])
        except (TypeError, ValueError):
            continue
        ps, ml, tot = o.get("pointSpread") or {}, o.get("moneyline") or {}, o.get("total") or {}
        g_ = lambda d, *p: num(_dig(d, *p))
        rows.append({
            "espn_id": gid, "kickoff": r["kickoff"], "division": r["division"],
            "home": r["home_name"], "away": r["away_name"],
            "hs": float(r["home_score"]), "as": float(r["away_score"]),
            "sp_h_open": g_(ps, "home", "open", "line"), "sp_h_close": g_(ps, "home", "close", "line"),
            "sp_h_odds": g_(ps, "home", "close", "odds"), "sp_a_odds": g_(ps, "away", "close", "odds"),
            "ml_h_open": g_(ml, "home", "open", "odds"), "ml_a_open": g_(ml, "away", "open", "odds"),
            "ml_h_close": g_(ml, "home", "close", "odds"), "ml_a_close": g_(ml, "away", "close", "odds"),
            "tot_open": g_(tot, "over", "open", "line"), "tot_close": g_(tot, "over", "close", "line"),
            "over_odds": g_(tot, "over", "close", "odds"), "under_odds": g_(tot, "under", "close", "odds"),
        })
    df = pd.DataFrame(rows)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    cut = df["kickoff"].quantile(0.5)
    df["half"] = np.where(df["kickoff"] <= cut, "early", "late")
    return df


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


# ------------------------------------------------------------------ bets (one row per side)

def ml_bets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["ml_h_open", "ml_a_open", "ml_h_close", "ml_a_close"])
    out = []
    for _, r in d.iterrows():
        if r.hs == r["as"]:
            continue
        po = implied(r.ml_h_open) / (implied(r.ml_h_open) + implied(r.ml_a_open))
        pc = implied(r.ml_h_close) / (implied(r.ml_h_close) + implied(r.ml_a_close))
        for side, odds, move, won in (("home", r.ml_h_close, pc - po, r.hs > r["as"]),
                                      ("away", r.ml_a_close, po - pc, r["as"] > r.hs)):
            out.append({"espn_id": r.espn_id, "half": r.half, "division": r.division, "side": side,
                        "odds": odds, "move": move, "won": won,
                        "profit": american_profit(odds) if won else -1.0})
    return pd.DataFrame(out)


def spread_bets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["sp_h_open", "sp_h_close"])
    out = []
    for _, r in d.iterrows():
        margin = r.hs - r["as"]
        for side, line, open_, odds, m in (("home", r.sp_h_close, r.sp_h_open, r.sp_h_odds, margin),
                                           ("away", -r.sp_h_close, -r.sp_h_open, r.sp_a_odds, -margin)):
            odds = -110.0 if np.isnan(odds) else odds
            res = m + line
            profit = 0.0 if res == 0 else (american_profit(odds) if res > 0 else -1.0)
            # move < 0: this side's line got shorter, i.e. the market moved toward it
            out.append({"espn_id": r.espn_id, "half": r.half, "division": r.division, "side": side,
                        "line": line, "move": line - open_, "odds": odds, "won": res > 0, "push": res == 0,
                        "profit": profit})
    return pd.DataFrame(out)


def total_bets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["tot_open", "tot_close"])
    out = []
    for _, r in d.iterrows():
        pts = r.hs + r["as"]
        for side, odds, sign in (("over", r.over_odds, 1), ("under", r.under_odds, -1)):
            odds = -110.0 if np.isnan(odds) else odds
            res = sign * (pts - r.tot_close)
            profit = 0.0 if res == 0 else (american_profit(odds) if res > 0 else -1.0)
            out.append({"espn_id": r.espn_id, "half": r.half, "division": r.division, "side": side,
                        "total": r.tot_close, "move": sign * (r.tot_close - r.tot_open),
                        "odds": odds, "won": res > 0, "push": res == 0, "profit": profit})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ summarise

def summarise(b: pd.DataFrame, band: str) -> pd.DataFrame:
    rows = []
    for k, d in b.groupby(band, observed=True):
        n = len(d)
        if n == 0:
            continue
        roi = d["profit"].mean()
        se = d["profit"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        e, l_ = d[d.half == "early"], d[d.half == "late"]
        rows.append({
            band: k, "bets": n, "W": int(d["won"].sum()), "L": int(n - d["won"].sum() - d.get("push", pd.Series(False, index=d.index)).sum()),
            "win%": round(100 * d["won"].mean(), 1), "ROI%": round(100 * roi, 1),
            "95% range": f"{100 * (roi - 1.96 * se):+.0f} to {100 * (roi + 1.96 * se):+.0f}" if n > 1 else "",
            "z": round(roi / se, 2) if se and se > 0 else np.nan,
            "early ROI%": round(100 * e["profit"].mean(), 1) if len(e) else np.nan,
            "late ROI%": round(100 * l_["profit"].mean(), 1) if len(l_) else np.nan,
            "early/late n": f"{len(e)}/{len(l_)}",
        })
    return pd.DataFrame(rows)


ML_PRICE = [-1e9, -1000, -400, -200, -130, 129.9, 199.9, 399.9, 999.9, 1e9]
ML_PRICE_L = ["<=-1000", "-999..-400", "-399..-200", "-199..-130", "-129..+129",
              "+130..+199", "+200..+399", "+400..+999", ">=+1000"]
PROB_MOVE = [-1, -0.08, -0.04, -0.015, 0.015, 0.04, 0.08, 1]
PROB_MOVE_L = ["drifted 8%+", "drifted 4-8%", "drifted 1.5-4%", "flat (±1.5%)",
               "steamed 1.5-4%", "steamed 4-8%", "steamed 8%+"]
# lines move in half points: edges sit between them so both sides of a move land in mirror bands
PT_MOVE = [-99, -2.75, -1.75, -0.75, -0.25, 0.25, 0.75, 1.75, 2.75, 99]
PT_MOVE_L = ["toward 3+", "toward 2-2.5", "toward 1-1.5", "toward 0.5", "no move",
             "away 0.5", "away 1-1.5", "away 2-2.5", "away 3+"]
SPREAD_SIZE = [-99, -21.25, -14.25, -7.25, -3.25, -0.25, 3.25, 7.25, 14.25, 21.25, 99]
SPREAD_SIZE_L = ["fav 21.5+", "fav 14.5-21", "fav 7.5-14", "fav 3.5-7", "fav 0.5-3",
                 "pk/dog 0-3", "dog 3.5-7", "dog 7.5-14", "dog 14.5-21", "dog 21.5+"]
TOT_MOVE = [-99, -2.75, -1.25, -0.25, 0.25, 1.25, 2.75, 99]
TOT_MOVE_L = ["against 3+", "against 1.5-2.5", "against 0.5-1", "no move",
              "with 0.5-1", "with 1.5-2.5", "with 3+"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--out", default="out/bands")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_games(Path(args.data_dir))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)

    ml, sp, tt = ml_bets(df), spread_bets(df), total_bets(df)
    ml["price"] = pd.cut(ml["odds"], ML_PRICE, labels=ML_PRICE_L)
    ml["move_band"] = pd.cut(ml["move"], PROB_MOVE, labels=PROB_MOVE_L)
    sp["move_band"] = pd.cut(sp["move"], PT_MOVE, labels=PT_MOVE_L)
    sp["size"] = pd.cut(sp["line"], SPREAD_SIZE, labels=SPREAD_SIZE_L)
    tt["move_band"] = pd.cut(tt["move"], TOT_MOVE, labels=TOT_MOVE_L)

    tables = {
        "moneyline_by_closing_price": summarise(ml, "price"),
        "moneyline_by_move (bet at close)": summarise(ml, "move_band"),
        "spread_by_move (bet at close)": summarise(sp, "move_band"),
        "spread_by_closing_size": summarise(sp, "size"),
        "totals_by_move (bet at close)": summarise(tt, "move_band"),
        "totals_over_vs_under": summarise(tt, "side"),
        "spread_home_vs_away": summarise(sp, "side"),
    }
    # price x move for the moneyline (where favourite-longshot and steam interact)
    ml["price_move"] = ml["price"].astype(str) + " | " + ml["move_band"].astype(str)
    pm = summarise(ml, "price_move")
    tables["moneyline_price_x_move (bets>=15)"] = pm[pm.bets >= 15]

    lines = [f"# Line-move and closing-price bands\n\nGames: {len(df)} finished with DraftKings open/close; "
             f"split at {df['kickoff'].quantile(0.5):%b %d} into early/late halves.\n"]
    for name, t in tables.items():
        t.to_csv(out / f"{name.split(' ')[0]}.csv", index=False)
        print(f"\n== {name}\n{t.to_string(index=False)}")
        lines.append(f"\n## {name}\n\n{t.to_markdown(index=False)}\n")
    n_tests = sum(len(t) for t in tables.values())
    lines.append(f"\n{n_tests} bands tested: at a 5% threshold about {n_tests * 0.05:.0f} would look "
                 f"significant by chance alone.\n")
    (out / "bands.md").write_text("".join(lines))
    print(f"\n{n_tests} bands tested (~{n_tests * 0.05:.0f} expected false positives at |z|>1.96)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
