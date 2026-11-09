"""Every dataset in one scan: which conditions predicted results in BOTH seasons?

  python analysis/full_scan.py --data-dir store/data [--perms 30] [--out reports/full_scan.md]

Two tables, one row per bet:
  spread sides   market features (line move, tickets, money, money-vs-tickets gap, moneyline
                 versions, favorite/dog, home/away, size, ranked, division)
                 + football (luck, success rate, turnovers, stats-fair line vs the real line)
                 + schedule spots (rest, last game's margin and ATS result, streaks, week, slot)
  totals sides   total move, tickets, money, gap, size, division, week, slot
                 + football (teams' scoring and success rates vs the posted total)

Every 1-, 2- and 3-feature combination is graded. A pattern "survives" when it covers
BAR%+ with MIN_N+ bets in each season separately. The same scan is rerun on coin-flipped
results (each game's outcome flipped at random) to count how many patterns survive by luck:
if the real count is not well above the shuffled count, the survivors are noise.
Survivors are then ranked by their WORSE season, with the family they belong to.
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
import backtest as bt  # noqa: E402
import bands  # noqa: E402
import edge_search as es  # noqa: E402
import explosive  # noqa: E402
import patterns as P  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402

# the store holds thousands of snapshot files: load each table once per run
_load, _games, _cache = explosive.load, bt.games, {}


def _cached_load(D, table):
    if table not in _cache:
        _cache[table] = _load(D, table)
    return _cache[table].copy()


def _cached_games(D):
    if "__games" not in _cache:
        _cache["__games"] = _games(D)
    return _cache["__games"].copy()


explosive.load, bt.games = _cached_load, _cached_games

MIN_N, BAR = 40, 55.0
SLOPE = 129.5
OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def cut(v, e, lab):
    return pd.cut(v, e, labels=lab).astype(object)


# ------------------------------------------------------------------ features

def team_context(D: Path) -> pd.DataFrame:
    """Per game and side, known before kickoff: season-to-date football stats and schedule spot."""
    t = P.team_stats(D, slope=SLOPE)
    t["tot"] = t.pf + t.pa
    grp = t.groupby(["team", "season"])
    t["pre_tot"] = grp.tot.transform(lambda s: s.shift().expanding().mean())
    t["pre_pf"] = grp.pf.transform(lambda s: s.shift().expanding().mean())
    t["pre_pa"] = grp.pa.transform(lambda s: s.shift().expanding().mean())
    t["pre_sr_off"] = grp.sr.transform(lambda s: s.shift().expanding().mean())
    t["pre_sr_def"] = grp.sr_against.transform(lambda s: s.shift().expanding().mean())
    t["prev_margin"] = grp.margin.shift()
    t["prev_ko"] = grp.kickoff.shift()
    t["rest"] = (t.kickoff - t.prev_ko).dt.days
    w = (t.margin > 0).astype(int)
    t["w"] = w
    t["streak"] = grp.w.transform(lambda s: s.shift().rolling(3).sum())     # wins in the last 3
    keep = ["espn_id", "side", "team", "season", "kickoff", "pre_n", "pre_luck", "pre_sr_diff", "pre_to_margin",
            "pre_tot", "pre_pf", "pre_pa", "pre_sr_off", "pre_sr_def", "prev_margin", "rest", "streak"]
    return t[keep]


def spread_table(D: Path) -> tuple[pd.DataFrame, list[str]]:
    s = es.side_rows(D, None)
    c = team_context(D)
    me = c.drop(columns=["team", "season", "kickoff"])
    opp = me.assign(side=np.where(me.side == "home", "away", "home")).add_prefix("o_").rename(
        columns={"o_espn_id": "espn_id", "o_side": "side"})
    s = s.merge(me, on=["espn_id", "side"], how="left").merge(opp, on=["espn_id", "side"], how="left")
    ok = (s.pre_n >= 3) & (s.o_pre_n >= 3)
    s["luck"] = cut((s.pre_luck - s.o_pre_luck).where(ok), [-99, -7, -3, 3, 7, 99],
                    ["unlucky7+", "unlucky3", "even", "lucky3", "lucky7+"])
    s["sr"] = cut((s.pre_sr_diff - s.o_pre_sr_diff).where(ok), [-9, -.08, -.03, .03, .08, 9],
                  ["sr--", "sr-", "even", "sr+", "sr++"])
    s["tov"] = cut((s.pre_to_margin - s.o_pre_to_margin).where(ok), [-99, -1, -.3, .3, 1, 99],
                   ["to--", "to-", "even", "to+", "to++"])
    hfa = np.where(s.side == "home", 2.5, -2.5)
    fair = -(SLOPE * (s.pre_sr_diff - s.o_pre_sr_diff) / 2 + hfa)
    s["value"] = cut((s.line - fair).where(ok), [-99, -6, -2.5, 2.5, 6, 99],
                     ["stats-6", "stats-3", "fair", "stats+3", "stats+6"])       # + = more points than stats say
    s["rest"] = np.select([s.rest.isna() | s.o_rest.isna(), s.rest - s.o_rest >= 5, s.o_rest - s.rest >= 5],
                          [None, "more rest", "less rest"], "same")
    s["prev"] = cut(s.prev_margin, [-999, -17, -1, 16, 999], ["lost big", "lost", "won", "won big"])
    s["o_prev"] = cut(s.o_prev_margin, [-999, -17, -1, 16, 999], ["opp lost big", "opp lost", "opp won", "opp won big"])
    s["form"] = np.select([s.streak.isna(), s.streak == 3, s.streak == 0], [None, "won 3 straight", "lost 3 straight"], "mixed")
    # last game's result against the spread, from this same table
    kick = bt.games(D).set_index("espn_id").kickoff
    s["kick"] = s.espn_id.map(kick)
    team = c.set_index(["espn_id", "side"]).team
    s["team"] = [team.get((e, sd)) for e, sd in zip(s.espn_id, s.side)]
    s = s.sort_values("kick")
    s["prev_ats"] = s.groupby(["team", "season"]).cover.shift().map({1: "covered last", -1: "failed last"})
    start = s.groupby("season").kick.transform("min")
    s["week"] = cut((s.kick - start).dt.days // 7 + 1, [0, 4, 8, 30], ["wk1-4", "wk5-8", "wk9+"])
    et = s.kick.dt.tz_convert("America/New_York")
    s["slot"] = np.select([et.dt.dayofweek != 5, et.dt.hour < 15, et.dt.hour < 19], ["weekday", "sat early", "sat mid"], "sat night")
    feats = ["sp_move", "ml_move", "sp_tix", "sp_mon", "sp_gap", "ml_tix", "ml_gap", "fav", "home", "size", "rank", "div",
             "luck", "sr", "tov", "value", "rest", "prev", "o_prev", "form", "prev_ats", "week", "slot"]
    s["w"], s["l"], s["p"] = (s.cover > 0).astype(int), (s.cover < 0).astype(int), s.sp_p
    return s.reset_index(drop=True), feats


def totals_table(D: Path) -> tuple[pd.DataFrame, list[str]]:
    f = explosive.load(D, "an_final_lines")
    f = f[f.market == "totals"]
    for c in ("point", "price", "money_pct", "tickets_pct"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    cons = f[f.book == "an_consensus"]
    op = f[f.book == "an_open"].drop_duplicates(["event_id", "side"], keep="last").set_index(["event_id", "side"]).point
    g = explosive.load(D, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last().reset_index()
    m = Matcher(g)
    R = bt.games(D).set_index("espn_id")
    ids = {}
    for eid, e in cons.groupby("event_id"):
        r = e.iloc[0]
        gid, _ = m.match(r.home_team, r.away_team, pd.Timestamp(r.commence_time), 12)
        if gid is not None and gid in R.index:
            ids[eid] = gid
    t = cons[cons.event_id.isin(ids)].drop_duplicates(["event_id", "side"], keep="last").copy()
    t["espn_id"] = t.event_id.map(ids)
    t = t.drop_duplicates(["espn_id", "side"], keep="last")
    t["season"], t["div"], t["kick"] = t.espn_id.map(R.season), t.espn_id.map(R.division), t.espn_id.map(R.kickoff)
    sg = np.where(t.side == "over", 1, -1)
    res = sg * (t.espn_id.map(R.hs) + t.espn_id.map(R["as"]) - t.point)
    t["cover"] = np.sign(res)
    mv = sg * (t.point - [op.get((e, s), np.nan) for e, s in zip(t.event_id, t.side)])
    ok = t.money_pct.between(3, 96)
    t["ou"] = t.side
    t["t_move"] = cut(mv, [-99, -1.75, -0.75, 0.75, 1.75, 99], ["away2+", "away1", "flat", "toward1", "toward2+"])
    t["t_tix"] = cut(t.tickets_pct.where(ok), [-1, 30, 45, 55, 70, 101], ["<30", "30-45", "45-55", "55-70", "70+"])
    t["t_mon"] = cut(t.money_pct.where(ok), [-1, 30, 45, 55, 70, 101], ["<30", "30-45", "45-55", "55-70", "70+"])
    t["t_gap"] = cut((t.money_pct - t.tickets_pct).where(ok), [-101, -10, -3, 3, 10, 101], ["m<<t", "m<t", "even", "m>t", "m>>t"])
    t["t_size"] = cut(t.point, [0, 44.5, 51.5, 58.5, 200], ["<45", "45-51", "52-58", "59+"])
    c = team_context(D)
    h, a = c[c.side == "home"].set_index("espn_id"), c[c.side == "away"].set_index("espn_id")
    okn = (t.espn_id.map(h.pre_n) >= 3) & (t.espn_id.map(a.pre_n) >= 3)
    stat_total = (t.espn_id.map(h.pre_pf) + t.espn_id.map(a.pre_pa) + t.espn_id.map(a.pre_pf) + t.espn_id.map(h.pre_pa)) / 2
    t["t_value"] = cut((sg * (stat_total - t.point)).where(okn), [-99, -7, -3, 3, 7, 99],
                       ["stats-7", "stats-3", "fair", "stats+3", "stats+7"])       # + = stats favor this side
    off = (t.espn_id.map(h.pre_sr_off) + t.espn_id.map(a.pre_sr_off)) / 2
    t["t_off"] = cut(off.where(okn), [0, .40, .45, 1], ["weak offenses", "avg offenses", "strong offenses"])
    start = t.groupby("season").kick.transform("min")
    t["week"] = cut((t.kick - start).dt.days // 7 + 1, [0, 4, 8, 30], ["wk1-4", "wk5-8", "wk9+"])
    et = t.kick.dt.tz_convert("America/New_York")
    t["slot"] = np.select([et.dt.dayofweek != 5, et.dt.hour < 15, et.dt.hour < 19], ["weekday", "sat early", "sat mid"], "sat night")
    t["w"], t["l"] = (t.cover > 0).astype(int), (t.cover < 0).astype(int)
    t["p"] = np.where(t.cover == 0, 0.0, np.where(t.cover > 0, [bands.american_profit(-110 if np.isnan(o) else o) for o in t.price], -1.0))
    feats = ["ou", "t_move", "t_tix", "t_mon", "t_gap", "t_size", "t_value", "t_off", "div", "week", "slot"]
    return t.reset_index(drop=True), feats


# ------------------------------------------------------------------ scan

class Scanner:
    """All 1..k feature combinations by integer codes and bincount (thousands of groupbys otherwise)."""

    def __init__(self, s: pd.DataFrame, feats: list[str], max_k: int):
        self.season = (s.season.values == 2026).astype(np.int64)            # 0 = 2025, 1 = 2026
        self.codes, self.labels = {}, {}
        for f in feats:
            c = pd.Categorical(s[f])
            self.codes[f], self.labels[f] = c.codes.astype(np.int64), [str(x) for x in c.categories]
        self.combos = [c for k in range(1, max_k + 1) for c in itertools.combinations(feats, k)]
        self.keys = []
        for combo in self.combos:
            key, size, ok = self.season.copy(), 2, np.ones(len(s), bool)
            for f in combo:
                n = len(self.labels[f])
                ok &= self.codes[f] >= 0
                key = key + size * np.where(self.codes[f] >= 0, self.codes[f], 0)
                size *= n
            self.keys.append((key[ok], ok, size))

    def cells(self, w: np.ndarray, l_: np.ndarray, p: np.ndarray | None = None):
        """Yield (combo index, per-cell arrays shaped [cells, 2 seasons]) for wins, losses, profit."""
        for i, (key, ok, size) in enumerate(self.keys):
            W = np.bincount(key, w[ok], size).reshape(-1, 2)
            L = np.bincount(key, l_[ok], size).reshape(-1, 2)
            Pr = np.bincount(key, p[ok], size).reshape(-1, 2) if p is not None else None
            yield i, W, L, Pr

    def name(self, i: int, cell: int) -> str:
        out = []
        for f in self.combos[i]:
            n = len(self.labels[f])
            out.append(self.labels[f][cell % n])
            cell //= n
        return " & ".join(out)


def passing(W: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = W + L
    c = 100 * W / np.maximum(n, 1)
    ok = (n >= MIN_N).all(axis=1) & (c >= BAR).all(axis=1)
    return ok, c.min(axis=1)


def survivors(sc: Scanner, w, l_, p) -> pd.DataFrame:
    rows = []
    for i, W, L, Pr in sc.cells(w, l_, p):
        ok, worst = passing(W, L)
        for cell in np.flatnonzero(ok):
            n = W[cell] + L[cell]
            rows.append({"combo": sc.name(i, cell), "k": len(sc.combos[i]), "w_2025": W[cell, 0], "l_2025": L[cell, 0],
                         "w_2026": W[cell, 1], "l_2026": L[cell, 1], "p_2025": Pr[cell, 0], "p_2026": Pr[cell, 1],
                         "n5": n[0], "n6": n[1], "c5": 100 * W[cell, 0] / n[0], "c6": 100 * W[cell, 1] / n[1],
                         "worst": worst[cell]})
    cols = ["combo", "k", "w_2025", "l_2025", "w_2026", "l_2026", "p_2025", "p_2026", "n5", "n6", "c5", "c6", "worst"]
    return pd.DataFrame(rows, columns=cols).sort_values("worst", ascending=False)


def null_counts(sc: Scanner, cover: np.ndarray, max_k: int) -> tuple[dict, float]:
    w, l_ = (cover > 0).astype(float), (cover < 0).astype(float)
    counts, best = {k: 0 for k in range(1, max_k + 1)}, 0.0
    for i, W, L, _ in sc.cells(w, l_):
        ok, worst = passing(W, L)
        if ok.any():
            counts[len(sc.combos[i])] += int(ok.sum())
            best = max(best, float(worst[ok].max()))
    return counts, best


def run(name: str, s: pd.DataFrame, feats: list[str], perms: int, max_k: int = 3) -> pd.DataFrame:
    say(f"\n## {name}")
    say(f"- bets {len(s)} | games by season {s.drop_duplicates('espn_id').season.value_counts().sort_index().to_dict()} | features {len(feats)}")
    sc = Scanner(s, feats, max_k)
    real = survivors(sc, s.w.values.astype(float), s.l.values.astype(float), s.p.fillna(0).values.astype(float))
    tested = sum(k[2] // 2 for k in sc.keys)
    rng = np.random.default_rng(7)
    games, inv = np.unique(s.espn_id.values, return_inverse=True)
    nulls = {k: [] for k in range(1, max_k + 1)}
    best = []
    for _ in range(perms):
        flip = rng.choice([-1, 1], len(games))[inv]
        counts, b = null_counts(sc, s.cover.values * flip, max_k)
        for k in nulls:
            nulls[k].append(counts[k])
        best.append(b)
    say(f"- patterns tested: {tested}; a survivor covers {BAR:.0f}%+ on {MIN_N}+ bets in EACH season")
    say("| features in the pattern | real survivors | by luck (shuffled mean) | luck 95th pct | verdict |")
    say("|---|---|---|---|---|")
    for k in nulls:
        r, nl = (real.k == k).sum(), np.array(nulls[k])
        verdict = "above luck" if r > np.percentile(nl, 95) else "no better than luck"
        say(f"| {k} | {r} | {nl.mean():.1f} | {np.percentile(nl, 95):.0f} | {verdict} |")
    say(f"- best survivor's worse season: real {real.worst.max() if len(real) else 0:.1f}% | shuffled best: mean "
        f"{np.mean(best):.1f}%, 95th pct {np.percentile(best, 95):.1f}%")
    say("\n| pattern | 2025 | 2026 |")
    say("|---|---|---|")
    for r in real.head(25).itertuples():
        say(f"| {r.combo} | {r.w_2025:.0f}-{r.l_2025:.0f} ({r.c5:.1f}%) ROI {100 * r.p_2025 / max(r.n5, 1):+.1f}% | "
            f"{r.w_2026:.0f}-{r.l_2026:.0f} ({r.c6:.1f}%) ROI {100 * r.p_2026 / max(r.n6, 1):+.1f}% |")
    # which single values show up most among survivors: the "families"
    if len(real):
        parts = pd.Series([x for c in real.combo for x in c.split(" & ")]).value_counts()
        say("\n- values that appear most among survivors: " + ", ".join(f"{k} ({v})" for k, v in parts.head(10).items()))
    return real


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    D = Path(a.data_dir)
    say(f"# Full scan ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d})")
    s, feats = spread_table(D)
    for f in feats:
        s[f] = s[f].where(s[f].notna(), None)
    run("Spreads: every market, football and schedule feature", s, feats, a.perms)
    t, tf = totals_table(D)
    run("Totals: over/under", t, tf, a.perms)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(OUT) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
