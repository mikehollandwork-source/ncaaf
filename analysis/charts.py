"""Charts: rule tracker and game-day board (matplotlib + seaborn, PNG).

  python analysis/charts.py tracker --data-dir store/data --out reports/charts
  python analysis/charts.py board   --data-dir store/data --out reports/charts [--hours 36]

tracker  cumulative units for each locked rule (graded from its logged snapshots) and for the
         "least money + line moved toward it" method backtested on 2025 vs 2026, each against
         break-even (0) with a band showing how far luck alone carries a -110 bettor (95%).
board    one PNG per upcoming game: the spread over time (DraftKings, Pinnacle, opener) above
         the share of money and bets on the home team (DraftKings and Action Network).

Colors follow the source in every chart: DraftKings blue, Pinnacle orange, Action Network aqua
(validated palette slots 1-3); lines carry direct labels because aqua is below 3:1 on white.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import bands  # noqa: E402
import explosive  # noqa: E402
import track_rule as tr  # noqa: E402
from ncaaf_tracker.matching import Matcher  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
DK, PIN, AN = "#2a78d6", "#eb6834", "#1baf7a"           # source colors, fixed everywhere
SEASON = {2025: "#2a78d6", 2026: "#eb6834"}
RULE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed slot order, never cycled
ET = ZoneInfo("America/New_York")


def style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 12,
        "font.size": 10, "lines.linewidth": 2, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False})


def label_end(ax, x, y, text, color, dy=0):
    """Queue a direct label at a line's end; place_labels() lays them out without overlaps."""
    ax.__dict__.setdefault("_end_labels", []).append((x, y, text, color))


def place_labels(ax, min_gap: float = 16) -> None:
    items = ax.__dict__.pop("_end_labels", [])
    if not items:
        return
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    pts = sorted(((to_px((matplotlib.dates.date2num(x) if isinstance(x, pd.Timestamp) else x, y))[1], i)
                  for i, (x, y, _, _) in enumerate(items)))
    placed = []
    for py, i in pts:                                   # bottom-up, push each label above the last
        py = max(py, placed[-1][0] + min_gap) if placed else py
        placed.append((py, i))
    top = ax.bbox.y1 - 8                                 # keep the stack inside the axes
    if placed and placed[-1][0] > top:
        shift = placed[-1][0] - top
        placed = [(py - shift, i) for py, i in placed]
    for py, i in placed:
        x, y, text, color = items[i]
        xd = matplotlib.dates.date2num(x) if isinstance(x, pd.Timestamp) else x
        orig = to_px((xd, y))[1]
        ax.annotate(text, (x, y), xytext=(6, py - orig), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold", annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=color, lw=1))


def luck_band(ax, n_max: int) -> None:
    """95% range of cumulative units for a bettor with no edge at -110 (win 52.4%, EV ~0)."""
    n = np.arange(0, n_max + 1)
    sd = 0.954 * np.sqrt(n)                      # per-bet sd of a -110 bet in units
    ax.fill_between(n, -1.96 * sd, 1.96 * sd, color=MUTED, alpha=0.15, lw=0, label="luck range (95%)")
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.text(0.01, -0.3, "break-even", color=INK2, fontsize=8, transform=ax.get_yaxis_transform(), va="top")


def profit(res: pd.Series, odds: pd.Series) -> np.ndarray:
    return np.array([0.0 if r == 0 else (bands.american_profit(-110 if pd.isna(o) else o) if r > 0 else -1.0)
                     for r, o in zip(res, odds)])


# ------------------------------------------------------------------ tracker

def live_rules(D: Path) -> dict[str, pd.DataFrame]:
    out = {}
    b = tr.grade(D, "rule_picks")
    if len(b):
        out["ranked_rlm_v1"] = b
    for rule in ("sharp_split_v1", "fade90_v1"):
        b = tr.grade(D, "split_picks", rule, tr.SPLIT_LOCKED)
        if len(b):
            out[rule] = b
    for k, b in out.items():
        b["pick_line"] = pd.to_numeric(b.pick_line, errors="coerce")
        b["pick_odds"] = pd.to_numeric(b.pick_odds, errors="coerce")
        b = b.sort_values("commence_time")
        b["units"] = profit(b.margin + b.pick_line, b.pick_odds).cumsum()
        out[k] = b
    return out


def method_sides(D: Path) -> pd.DataFrame:
    import backtest as bt
    import edge_search as es
    g = bt.games(D)
    g = g[g.sp_close.notna() & g.sp_open.notna()]
    sp = es.splits(D)
    rows = []
    for side, sg in (("home", 1), ("away", -1)):
        x = g.merge(sp[sp.side == side], on="espn_id")
        rows.append(pd.DataFrame({
            "season": x.season, "kickoff": x.kickoff, "mon": x.sp_mon,
            "res": sg * (x.hs - x["as"]) + (x.sp_close if side == "home" else -x.sp_close),
            "odds": x.sp_h_odds if side == "home" else x.sp_a_odds,
            "move": -sg * (x.sp_close - x.sp_open)}))
    s = pd.concat(rows)
    return s[s.mon.between(3, 96) & s.res.notna()].sort_values("kickoff")


def method_backtest(sides: pd.DataFrame, move: float, cap: float) -> pd.DataFrame:
    s = sides[(sides.move >= move) & (sides.mon < cap)].copy()
    s["p"] = profit(s.res, s.odds)
    return s


def tracker(D: Path, out: Path) -> list[Path]:
    style()
    paths = []
    rules = live_rules(D)
    fig, ax = plt.subplots(figsize=(9, 5))
    n_max = max([len(b) for b in rules.values()] + [10])
    luck_band(ax, n_max)
    for i, (name, b) in enumerate(rules.items()):
        x = np.arange(1, len(b) + 1)
        c = RULE_COLORS[i]
        ax.plot(np.r_[0, x], np.r_[0, b.units], color=c, label=name, marker="o", ms=4, mec=SURFACE, mew=1)
        w, l_ = int((b.margin + b.pick_line > 0).sum()), int((b.margin + b.pick_line < 0).sum())
        label_end(ax, x[-1], b.units.iloc[-1], f"{name}  {w}-{l_}  {b.units.iloc[-1]:+.1f}u", c)
    ax.set_xlim(0, n_max * 1.35)
    ax.set_xlabel("bets graded since the rule was locked")
    ax.set_ylabel("units won (1 unit per bet)")
    ax.set_title("Locked rules, live record vs luck", loc="left")
    place_labels(ax)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = out / "tracker_live_rules.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    sides = method_sides(D)
    for move, cap in ((1, 50), (2, 30)):
        s = method_backtest(sides, move, cap)
        fig, ax = plt.subplots(figsize=(9, 5))
        luck_band(ax, int(s.groupby("season").size().max()))
        for season, d in s.groupby("season"):
            u = d.p.cumsum().values
            x = np.arange(1, len(u) + 1)
            w, l_ = int((d.res > 0).sum()), int((d.res < 0).sum())
            ax.plot(np.r_[0, x], np.r_[0, u], color=SEASON.get(int(season), INK2), label=str(season))
            label_end(ax, x[-1], u[-1], f"{season}: {w}-{l_} ({100 * w / max(w + l_, 1):.0f}%)  {u[-1]:+.1f}u",
                      SEASON.get(int(season), INK2))
        ax.set_xlim(0, s.groupby("season").size().max() * 1.4)
        ax.set_xlabel("bets, in kickoff order within the season")
        ax.set_ylabel("units won")
        ax.set_title(f"Least money + line moved toward it (move ≥{move} pt, money <{cap}%), by season",
                     loc="left")
        ax.legend(loc="upper left", fontsize=8)
        place_labels(ax)
        fig.tight_layout()
        p = out / f"tracker_method_move{move}_money{cap}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ board

def dk_kickoff(text: str, year: int):
    m = re.search(r"(\d{1,2})/(\d{1,2})\D+?(\d{1,2}):(\d{2})\s*([AP]M)", text or "")
    if not m:
        return pd.NaT
    mo, d, h, mi, ap = m.groups()
    h = int(h) % 12 + (12 if ap == "PM" else 0)
    return pd.Timestamp(year, int(mo), int(d), h, int(mi), tz=ET).tz_convert("UTC")


def resolve(df: pd.DataFrame, m: Matcher, when_col: str) -> pd.DataFrame:
    keys = df[["home_team", "away_team", when_col]].drop_duplicates()
    ids, flip = {}, {}
    for _, k in keys.iterrows():
        gid, sides = m.match(k.home_team, k.away_team, k[when_col], 12)
        if gid is not None:
            ids[(k.home_team, k.away_team)] = gid
            flip[(k.home_team, k.away_team)] = sides[k.home_team] != "home"
    key = list(zip(df.home_team, df.away_team))
    df = df.assign(espn_id=[ids.get(x) for x in key], flipped=[flip.get(x, False) for x in key])
    return df[df.espn_id.notna()]


def board(D: Path, out: Path, hours: float) -> list[Path]:
    style()
    now = pd.Timestamp.now(tz="UTC")
    g = explosive.load(D, "espn_games").sort_values("snapshot_ts").groupby("espn_id").last().reset_index()
    g["ko"] = pd.to_datetime(g.kickoff, utc=True)
    g = g[(g.ko > now) & (g.ko <= now + pd.Timedelta(hours=hours))]
    if g.empty:
        print("no games in the window")
        return []
    m = Matcher(g)
    lo = (now - pd.Timedelta(days=10)).isoformat()

    def lines(table, books):
        o = explosive.load(D, table)
        o = o[(o.market == "spreads") & o.book.isin(books) & (o.snapshot_ts >= lo)].copy()
        o["point"] = pd.to_numeric(o.point, errors="coerce")
        o = resolve(o, m, "commence_time")
        # home-team (ESPN home) spread
        home_name = np.where(o.flipped, o.away_team, o.home_team)
        o = o[o.outcome == home_name]
        o["t"] = pd.to_datetime(o.snapshot_ts, utc=True)
        return o[["espn_id", "book", "t", "point"]]

    ln = pd.concat([lines("an_odds", ["draftkings", "an_open"]), lines("pinnacle_odds", ["pinnacle"])])
    sp = explosive.load(D, "an_splits")
    sp = sp[(sp.market == "spreads") & (sp.snapshot_ts >= lo)].copy()
    sp = resolve(sp, m, "commence_time")
    sp = sp[(sp.side == "home") != sp.flipped]                  # rows for the ESPN home team
    for c in ("tickets_pct", "money_pct"):
        sp[c] = pd.to_numeric(sp[c], errors="coerce")
    sp = sp[sp.money_pct.between(3, 96)]
    sp["t"] = pd.to_datetime(sp.snapshot_ts, utc=True)
    dk = explosive.load(D, "dk_splits")
    if len(dk):
        dk = dk[(dk.market == "spread") & (dk.snapshot_ts >= lo)].copy()
        dk["ko"] = [dk_kickoff(x, now.year) for x in dk.game_time_et]
        dk = resolve(dk, m, "ko")
        dk = dk[(dk.side == "home") != dk.flipped]
        for c in ("handle_pct", "bets_pct"):
            dk[c] = pd.to_numeric(dk[c], errors="coerce")
        dk["t"] = pd.to_datetime(dk.snapshot_ts, utc=True)

    paths = []
    for _, game in g.sort_values("ko").iterrows():
        gid, ko = game.espn_id, game.ko
        L = ln[(ln.espn_id == gid) & (ln.t < ko)].sort_values("t")
        if L.empty:
            continue
        S = sp[(sp.espn_id == gid) & (sp.t < ko)].sort_values("t")
        K = dk[(dk.espn_id == gid) & (dk.t < ko)].sort_values("t") if len(dk) else pd.DataFrame()
        home = game.get("home_short") or game.home_name
        away = game.get("away_short") or game.away_name
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})

        for book, c, lab in (("draftkings", DK, "DraftKings"), ("pinnacle", PIN, "Pinnacle")):
            x = L[L.book == book]
            if len(x):
                a1.step(x.t, x.point, where="post", color=c, label=lab)
                label_end(a1, x.t.iloc[-1], x.point.iloc[-1], f"{lab} {x.point.iloc[-1]:+g}", c)
        op = L[L.book == "an_open"].point.dropna()
        if len(op):
            a1.axhline(op.iloc[-1], color=MUTED, ls=":", lw=1.5, label=f"opener {op.iloc[-1]:+g}")
        a1.set_ylabel(f"{home} spread\n(up = moving toward {home})", fontsize=9)
        a1.invert_yaxis()                                         # up = market rates the home team higher
        a1.legend(loc="best", fontsize=8, ncol=3)

        a2.axhline(50, color=INK2, lw=1, ls="--")
        for d, col_m, col_t, c, lab in ((K, "handle_pct", "bets_pct", DK, "DraftKings"),
                                        (S, "money_pct", "tickets_pct", AN, "Action Network")):
            if len(d):
                a2.plot(d.t, d[col_m], color=c, marker="o", ms=4, mec=SURFACE, mew=1, label=f"{lab} money %")
                a2.plot(d.t, d[col_t], color=c, ls="--", lw=1.5, label=f"{lab} bets %")
                label_end(a2, d.t.iloc[-1], d[col_m].iloc[-1], f"{lab} money {d[col_m].iloc[-1]:.0f}%", c)
        a2.set_ylim(0, 100)
        a2.set_ylabel(f"% on {home}", fontsize=9)
        a2.legend(loc="best", fontsize=8, ncol=2)

        # verdict line: did the line move against where the money is?
        dkl = L[L.book == "draftkings"].point.dropna()
        verdict = ""
        if len(dkl) and (len(S) or len(K)):
            mv = dkl.iloc[-1] - dkl.iloc[0]                        # < 0: moved toward home
            money = (K.handle_pct.iloc[-1] if len(K) else S.money_pct.iloc[-1])
            if abs(mv) >= 1:
                toward = home if mv < 0 else away
                money_on = money if toward == home else 100 - money
                verdict = (f"DraftKings moved {abs(mv):g} toward {toward}, {money_on:.0f}% of money on {toward}: "
                           + ("reverse line move" if money_on < 50 else "money agrees with the move"))
            else:
                verdict = f"Line flat (DraftKings moved {mv:+g})"
        fig.suptitle(f"{away} @ {home}  ·  {ko.tz_convert(ET):%a %b %-d, %-I:%M %p} ET", x=0.01, ha="left",
                     fontweight="bold", fontsize=13)
        a1.set_title(verdict, loc="left", fontsize=10, color=INK2, fontweight="normal", wrap=True)
        last = max(L.t.max(), S.t.max() if len(S) else L.t.max(), K.t.max() if len(K) else L.t.max())
        a2.set_xlim(L.t.min(), last + (last - L.t.min()) * 0.3)
        a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %-d", tz=ET))
        fig.tight_layout()
        place_labels(a1)
        place_labels(a2)
        slug = re.sub(r"[^a-z0-9]+", "-", f"{ko:%m%d}-{away}-at-{home}".lower()).strip("-")
        p = out / f"board_{slug}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tracker", "board"))
    ap.add_argument("--data-dir", default="store/data")
    ap.add_argument("--out", default="reports/charts")
    ap.add_argument("--hours", type=float, default=36)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = tracker(Path(a.data_dir), out) if a.what == "tracker" else board(Path(a.data_dir), out, a.hours)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
