"""Ranked matchups: how did past ones go, and what do this week's look like?

  python analysis/ranked.py --data-dir store/data

Ranked = AP top 25 that week (ESPN curatedRank). Two groups:
  ranked vs ranked   both teams in the top 25
  one ranked         a ranked team against an unranked one

Past games: closing DraftKings line vs result (favorite win rate, cover rate,
Brier), plus whether the team with the better pre-game explosive margin or
clutch WPA (earlier games only) won / covered.
Upcoming games: market win probability from the latest Action Network
consensus moneyline, with each team's season explosive and clutch numbers.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import bands  # noqa: E402
import explosive  # noqa: E402


def latest_games(data_dir: Path) -> pd.DataFrame:
    g = explosive.load(data_dir, "espn_games").sort_values("snapshot_ts")
    last = g.groupby("espn_id").last().reset_index()
    for c in ("home_rank", "away_rank"):
        last[c] = pd.to_numeric(last.get(c), errors="coerce")
    last["kickoff"] = pd.to_datetime(last["kickoff"], utc=True)
    last["n_ranked"] = last[["home_rank", "away_rank"]].notna().sum(axis=1)
    return last


def novig(h, a):
    ph, pa = bands.implied(h), bands.implied(a)
    return ph / (ph + pa)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 30)

    g = latest_games(data_dir)
    lines = bands.load_games(data_dir).set_index("espn_id")
    tg = explosive.team_games(data_dir)
    pre = explosive.pregame(tg, min_games=1)

    # ---------------------------------------------------------------- past ranked games
    fin = g[(g.completed == "True") & (g.n_ranked >= 1)].set_index("espn_id")
    fin = fin.join(lines[["sp_h_close", "ml_h_close", "ml_a_close", "sp_h_odds", "sp_a_odds"]], how="left")
    fin["hs"], fin["as"] = fin.home_score.astype(float), fin.away_score.astype(float)
    fin = fin[fin.hs != fin["as"]]
    fin["home_won"] = fin.hs > fin["as"]
    fin["p_home"] = [novig(h, a) if not (np.isnan(h) or np.isnan(a)) else np.nan
                     for h, a in zip(fin.ml_h_close, fin.ml_a_close)]
    ph = pre[pre.side == "home"].set_index("espn_id")[["pre_expl_margin", "pre_clutch_wpa", "pre_n"]]
    pa_ = pre[pre.side == "away"].set_index("espn_id")[["pre_expl_margin", "pre_clutch_wpa", "pre_n"]]
    fin = fin.join(ph.add_suffix("_h")).join(pa_.add_suffix("_a"))

    print("== Past ranked matchups (closing DraftKings line)")
    for label, sub in (("ranked vs ranked", fin[fin.n_ranked == 2]), ("one ranked", fin[fin.n_ranked == 1]),
                       ("any ranked", fin)):
        s = sub.dropna(subset=["p_home"])
        fav_home = s.p_home >= 0.5
        fav_won = (fav_home == s.home_won)
        brier = ((s.p_home - s.home_won) ** 2).mean()
        res = (s.hs - s["as"]) + s.sp_h_close
        fav_cover = np.where(s.sp_h_close <= 0, res > 0, res < 0)[res != 0]
        print(f"  {label:17} games {len(sub):3d} | favorite won {fav_won.mean():.0%} "
              f"(line expected {np.where(fav_home, s.p_home, 1 - s.p_home).mean():.0%}) | "
              f"favorite covered {fav_cover.mean():.0%} | Brier {brier:.3f}")
        # ranked team vs unranked: did the ranked team win?
        if label == "one ranked":
            ranked_home = sub.home_rank.notna()
            print(f"  {'':17} ranked team won {(ranked_home == sub.home_won).mean():.0%} of {len(sub)}")

    rr = fin[fin.n_ranked == 2].sort_values("kickoff")
    if len(rr):
        print("\n  ranked vs ranked results:")
        for _, r in rr.iterrows():
            fav = r.home_name if r.p_home >= 0.5 else r.away_name
            winner = r.home_name if r.home_won else r.away_name
            pf = max(r.p_home, 1 - r.p_home) if not np.isnan(r.p_home) else np.nan
            ex = (r.pre_expl_margin_h - r.pre_expl_margin_a) if not np.isnan(r.get("pre_expl_margin_h", np.nan)) else np.nan
            print(f"   {r.kickoff:%m-%d} #{int(r.away_rank)} {r.away_name[:22]:22} {int(r['as']):>2} @ "
                  f"#{int(r.home_rank)} {r.home_name[:22]:22} {int(r.hs):>2} | line fav {fav[:16]:16} {pf:.0%} | "
                  f"{'FAV' if fav == winner else 'UPSET':5} | home-away explosive margin edge {ex:+.1f}")

    for f, name in (("pre_expl_margin", "explosive margin"), ("pre_clutch_wpa", "clutch WPA")):
        s = fin.dropna(subset=[f + "_h", f + "_a"])
        s = s[s[f + "_h"] != s[f + "_a"]]
        better_home = s[f + "_h"] > s[f + "_a"]
        res = (s.hs - s["as"]) + s.sp_h_close
        cover = np.where(better_home, res > 0, res < 0)[res != 0]
        print(f"\n  better pre-game {name}: won {(better_home == s.home_won).mean():.0%}, "
              f"covered {cover.mean():.0%} (n={len(s)} ranked games)")

    # ---------------------------------------------------------------- upcoming ranked games
    now = pd.Timestamp.now(tz="UTC")
    up = g[(g.kickoff > now) & (g.kickoff < now + pd.Timedelta(days=8)) & (g.n_ranked >= 1)].copy()
    an = explosive.load(data_dir, "an_odds")
    season = tg.groupby("team")[["expl_margin", "clutch_wpa", "expl_rate", "expl_rate_against"]].mean()
    games_played = tg.groupby("team").size()
    rows = []
    if not an.empty:
        an = an[(an.book == "an_consensus") & (an.market == "h2h")].sort_values("snapshot_ts")
        an["price"] = pd.to_numeric(an.price, errors="coerce")
        lastp = an.groupby(["home_team", "away_team", "outcome"]).price.last()
    from ncaaf_tracker.matching import name_score, norm  # noqa: E402
    for _, r in up.sort_values("kickoff").iterrows():
        p_home = np.nan
        if not an.empty:
            for (ht, at), grp in lastp.groupby(level=[0, 1]):
                if min(name_score(ht, [norm(r.home_name)]), name_score(at, [norm(r.away_name)])) > 0.85:
                    try:
                        p_home = novig(grp.loc[(ht, at, ht)], grp.loc[(ht, at, at)])
                    except KeyError:
                        pass
                    break
        hm = season.reindex([r.home_id]).iloc[0]
        aw = season.reindex([r.away_id]).iloc[0]
        rows.append({
            "kickoff": f"{r.kickoff:%a %m-%d %H:%M}Z",
            "matchup": f"{'#%d ' % r.away_rank if not np.isnan(r.away_rank) else ''}{r.away_name} @ "
                       f"{'#%d ' % r.home_rank if not np.isnan(r.home_rank) else ''}{r.home_name}",
            "both_ranked": r.n_ranked == 2,
            "market_pick": (r.home_name if p_home >= 0.5 else r.away_name) if not np.isnan(p_home) else "",
            "win_prob": round(max(p_home, 1 - p_home), 3) if not np.isnan(p_home) else np.nan,
            "expl_margin_h": round(hm.expl_margin, 1), "expl_margin_a": round(aw.expl_margin, 1),
            "clutch_h": round(hm.clutch_wpa, 2), "clutch_a": round(aw.clutch_wpa, 2),
            "games_h": int(games_played.get(r.home_id, 0)), "games_a": int(games_played.get(r.away_id, 0)),
        })
    out = pd.DataFrame(rows)
    print(f"\n== Upcoming ranked matchups (next 8 days): {len(out)}")
    if len(out):
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
