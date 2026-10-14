"""Turn collected snapshots into analysis tables + a correlation report.

  python -m ncaaf_tracker.analyze --data-dir data --out out

Outputs (in --out):
  games_summary.csv    one row per game: result, ESPN open/close, tracked open/close
                       (consensus of all books), Kalshi/Polymarket open/close prob & volume
  book_open_close.csv  per game x book x market: first-seen line, closing line, #moves
  line_history.csv     every tracked book line snapshot (home spread, total, home win prob)
  hourly_panel.csv     game x hour (pre-kickoff only): consensus book line + prediction
                       market prob, volume, net taker flow ($) and book imbalance
  unmatched.csv        source games that couldn't be tied to an ESPN game (fix via aliases.csv)
  report.md            correlations, lead/lag, big-move agreement, closing-line accuracy

Conventions: spreads are the HOME team's line (negative = home favored);
probabilities are P(home wins), vig removed; flow > 0 means money bet on home.
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .matching import Matcher, load_aliases
from .sources.polymarket import is_moneyline

log = logging.getLogger("ncaaf")
SHARP_BOOK = "pinnacle"


# ------------------------------------------------------------------ loading

def load(data_dir: Path, table: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / table / "*" / "*.csv.gz")))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f, dtype=str) for f in files), ignore_index=True)
    return df


def num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def american_to_prob(a: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def mid(bid: pd.Series, ask: pd.Series, last: pd.Series) -> pd.Series:
    ok = bid.notna() & ask.notna() & (ask > bid) & (bid > 0) & (ask < 1)
    return pd.Series(np.where(ok, (bid + ask) / 2, last), index=bid.index)


# ------------------------------------------------------------------ games

def build_games(data_dir: Path) -> pd.DataFrame:
    g = load(data_dir, "espn_games")
    if g.empty:
        return g
    g["snapshot_dt"] = ts(g["snapshot_ts"])
    g = g.sort_values("snapshot_dt")
    open_cols = [c for c in g if c.startswith("espn_") and c.endswith("_open")]
    first_open = g.groupby("espn_id")[open_cols].first()          # first non-null seen
    latest = g.groupby("espn_id").last()                          # last non-null of each col
    latest[open_cols] = first_open
    latest = latest.reset_index()
    latest["kickoff_dt"] = ts(latest["kickoff"])
    num(latest, ["home_score", "away_score"] + [c for c in latest if c.startswith("espn_")
                                                 and c != "espn_id" and not c.endswith(("provider", "details", "raw"))])
    final = latest["status_state"].eq("post") | latest["completed"].astype(str).str.lower().eq("true")
    latest["final"] = final
    latest.loc[~final, ["home_score", "away_score"]] = np.nan   # ESPN reports 0-0 before kickoff
    latest["margin_home"] = np.where(final, latest["home_score"] - latest["away_score"], np.nan)
    latest["winner"] = np.select([final & (latest["margin_home"] > 0), final & (latest["margin_home"] < 0)],
                                 [latest["home_name"], latest["away_name"]], default=None)
    latest["home_won"] = np.where(final, (latest["margin_home"] > 0).astype(float), np.nan)
    latest["total_points"] = latest["home_score"] + latest["away_score"]
    return latest.drop(columns=["snapshot_dt", "snapshot_ts"], errors="ignore")


# ------------------------------------------------------------------ sportsbooks

NON_CONSENSUS_BOOKS = {"an_consensus", "an_open"}
HISTORY_HOURS = 72   # hourly grid built from exact line history for the last N hours pre-kickoff


def history_snapshots(h: pd.DataFrame, hours: int = HISTORY_HOURS) -> pd.DataFrame:
    """Turn change-by-change line history into snapshots comparable with polled data:
    the opener, every hour for the last `hours` before kickoff, and 1 min before kickoff."""
    if h.empty:
        return h
    h = h.copy()
    h["t"] = ts(h["updated_at"])
    h["ko"] = ts(h["commence_time"])
    h = h[h["t"].notna() & (h["t"] < h["ko"])].sort_values("t")
    grids = []
    for eid, d in h.groupby("event_id"):
        ko, first = d["ko"].iloc[0], d["t"].min()
        pts = pd.date_range(max(first, ko - pd.Timedelta(hours=hours)).ceil("h"), ko, freq="h", inclusive="left")
        # each book's own opener (sides post ms apart, so +5s), plus the close
        openers = d.groupby("book")["t"].min() + pd.Timedelta(seconds=5)
        pts = pts.append(pd.DatetimeIndex(list(openers) + [ko - pd.Timedelta(minutes=1)]))
        grids.append(pd.DataFrame({"event_id": eid, "snap": pts.unique().sort_values()}))
    grid = pd.concat(grids, ignore_index=True)
    series = h[["event_id", "book", "market", "outcome"]].drop_duplicates()
    grid = grid.merge(series, on="event_id").sort_values("snap")
    out = pd.merge_asof(grid, h.rename(columns={"t": "snap"}).sort_values("snap"), on="snap",
                        by=["event_id", "book", "market", "outcome"], direction="backward")
    out = out.dropna(subset=["price"])
    out["snapshot_ts"] = out["snap"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.drop(columns=["snap", "ko"], errors="ignore")


def load_book_rows(data_dir: Path) -> pd.DataFrame:
    """Every sportsbook source in one frame: Odds API (optional), Action Network
    live snapshots, Pinnacle, and Action Network exact history."""
    frames = []
    o = load(data_dir, "odds")
    if not o.empty:
        frames.append(o.rename(columns={"oddsapi_id": "event_id"}).assign(source="oddsapi"))
    for t in ("an_odds", "pinnacle_odds"):
        f = load(data_dir, t)
        if not f.empty:
            frames.append(f)
    hist = load(data_dir, "an_history")
    if not hist.empty:
        hist = hist.drop_duplicates(["event_id", "book", "market", "outcome", "updated_at"])
        frames.append(history_snapshots(hist))
    if not frames:
        return pd.DataFrame()
    cols = ["event_id", "snapshot_ts", "commence_time", "home_team", "away_team", "book", "market",
            "outcome", "price", "point", "source"]
    return pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)


def map_events(o: pd.DataFrame, matcher: Matcher, unmatched: list) -> dict:
    """event_id -> (espn_id, source home == ESPN home)."""
    mapping = {}
    ev = o.drop_duplicates("event_id")[["event_id", "home_team", "away_team", "commence_time", "source"]]
    for _, e in ev.iterrows():
        gid, sides = matcher.match(e["home_team"], e["away_team"], ts(pd.Series([e["commence_time"]]))[0], 12)
        if gid is None:
            unmatched.append({"source": e["source"], "id": e["event_id"], "team_a": e["home_team"],
                              "team_b": e["away_team"], "time": e["commence_time"]})
            continue
        mapping[e["event_id"]] = (gid, sides[e["home_team"]] == "home")
    return mapping


def build_book_lines(data_dir: Path, games: pd.DataFrame, matcher: Matcher, unmatched: list):
    o = load_book_rows(data_dir)
    if o.empty:
        return pd.DataFrame(), {}, set()
    num(o, ["price", "point"])
    o["snap"] = ts(o["snapshot_ts"])
    mapping = map_events(o, matcher, unmatched)
    o = o[o["event_id"].isin(mapping)].copy()
    o["espn_id"] = o["event_id"].map(lambda k: mapping[k][0])
    o["aligned"] = o["event_id"].map(lambda k: mapping[k][1])
    history_ids = set(o.loc[o["source"] == "an_history", "espn_id"])
    # which ESPN side is this outcome?
    is_src_home = o["outcome"] == o["home_team"]
    o["side"] = np.where(o["market"] == "totals", o["outcome"].str.lower(),
                         np.where(is_src_home == o["aligned"], "home", "away"))
    o = o.merge(games[["espn_id", "kickoff_dt"]], on="espn_id")
    o = o[o["snap"] < o["kickoff_dt"]]
    key = ["espn_id", "snap", "book"]
    h2h = o[o["market"] == "h2h"].pivot_table(index=key, columns="side", values="price", aggfunc="last")
    spr = o[(o["market"] == "spreads") & (o["side"] == "home")].groupby(key)[["point", "price"]].last()
    tot = o[(o["market"] == "totals") & (o["side"] == "over")].groupby(key)[["point", "price"]].last()
    out = pd.DataFrame(index=h2h.index.union(spr.index).union(tot.index))
    if {"home", "away"} <= set(h2h.columns):
        ph, pa = american_to_prob(h2h["home"]), american_to_prob(h2h["away"])
        out["ml_home"], out["ml_away"] = h2h["home"], h2h["away"]
        out["home_prob"] = pd.Series(ph / (ph + pa), index=h2h.index)
    out["spread_home"], out["spread_price"] = spr["point"], spr["price"]
    out["total"], out["total_over_price"] = tot["point"], tot["price"]
    return out.reset_index(), mapping, history_ids


def book_open_close(lines: pd.DataFrame, first_run, history_ids: set = frozenset()) -> pd.DataFrame:
    if lines.empty:
        return lines
    rows = []
    for (gid, book), d in lines.sort_values("snap").groupby(["espn_id", "book"]):
        for col in ("spread_home", "total", "ml_home", "home_prob"):
            s = d[["snap", col]].dropna()
            if s.empty:
                continue
            rows.append({
                "espn_id": gid, "book": book, "market": col,
                "open_ts": s["snap"].iloc[0], "open": s[col].iloc[0],
                "close_ts": s["snap"].iloc[-1], "close": s[col].iloc[-1],
                "move": s[col].iloc[-1] - s[col].iloc[0],
                "n_obs": len(s), "n_changes": int((s[col].diff().fillna(0) != 0).sum()),
                # Line already existed when tracking began -> true opener unknown.
                "opener_censored": (book != "an_open" and gid not in history_ids
                                    and s["snap"].iloc[0] <= first_run + pd.Timedelta(hours=1)),
            })
    return pd.DataFrame(rows)


def consensus(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return lines
    real = lines[~lines["book"].isin(NON_CONSENSUS_BOOKS)]
    c = real.groupby(["espn_id", "snap"]).agg(
        book_home_prob=("home_prob", "median"), book_spread=("spread_home", "median"),
        book_total=("total", "median"), n_books=("book", "nunique")).reset_index()
    sharp = lines[lines["book"] == SHARP_BOOK][["espn_id", "snap", "home_prob", "spread_home"]]
    sharp = sharp.rename(columns={"home_prob": "sharp_home_prob", "spread_home": "sharp_spread"})
    return c.merge(sharp, on=["espn_id", "snap"], how="left")


def build_splits(data_dir: Path, mapping: dict) -> pd.DataFrame:
    """Action Network public betting splits -> % of money / tickets on the ESPN home side."""
    sp = load(data_dir, "an_splits")
    if sp.empty or not mapping:
        return pd.DataFrame()
    num(sp, ["tickets_pct", "money_pct"])
    sp = sp[sp["event_id"].isin(mapping) & sp["side"].isin(["home", "away"])].copy()
    sp["espn_id"] = sp["event_id"].map(lambda k: mapping[k][0])
    aligned = sp["event_id"].map(lambda k: mapping[k][1])
    sp["espn_side"] = np.where((sp["side"] == "home") == aligned, "home", "away")
    sp = sp[sp["espn_side"] == "home"]
    sp["snap"] = ts(sp["snapshot_ts"])
    p = sp.pivot_table(index=["espn_id", "snap"], columns="market", values=["money_pct", "tickets_pct"],
                       aggfunc="last")
    out = pd.DataFrame(index=p.index)
    if ("money_pct", "spreads") in p:
        out["an_money_home_pct"] = p[("money_pct", "spreads")]
        out["an_tickets_home_pct"] = p[("tickets_pct", "spreads")]
    if ("money_pct", "h2h") in p:
        out["an_ml_money_home_pct"] = p[("money_pct", "h2h")]
    return out.reset_index()


# ------------------------------------------------------------------ Kalshi

def kalshi_ticker_sides(m: pd.DataFrame, matcher: Matcher, unmatched: list) -> dict[str, tuple[str, str]]:
    """Kalshi game-winner ticker -> (espn_id, 'home'|'away' team the ticker's yes pays on)."""
    tick_side: dict[str, tuple[str, str]] = {}
    for ev_t, d in m.groupby("event_ticker"):
        teams = d.drop_duplicates("ticker")[["ticker", "yes_sub_title"]].dropna()
        if len(teams) != 2:
            continue
        a, b = teams["yes_sub_title"].tolist()
        ko = ts(d["kickoff_est"]).dropna()
        when = ko.iloc[-1] if len(ko) else ts(d["ticker_date"]).iloc[0] + pd.Timedelta(hours=18)
        gid, sides = matcher.match(a, b, when, 14 if len(ko) == 0 else 12)
        if gid is None:
            unmatched.append({"source": "kalshi", "id": ev_t, "team_a": a, "team_b": b, "time": str(when)})
            continue
        for t, name in zip(teams["ticker"], teams["yes_sub_title"]):
            tick_side[t] = (gid, sides[name])
    return tick_side


def build_kalshi(data_dir: Path, games: pd.DataFrame, matcher: Matcher, unmatched: list):
    m = load(data_dir, "kalshi_markets")
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = m[m["series"] == "KXNCAAFGAME"].copy()
    num(m, ["yes_bid", "yes_ask", "last_price", "volume", "open_interest"])
    m["snap"] = ts(m["snapshot_ts"])
    tick_side = kalshi_ticker_sides(m, matcher, unmatched)
    m = m[m["ticker"].isin(tick_side)].copy()
    m["espn_id"] = m["ticker"].map(lambda t: tick_side[t][0])
    m["side"] = m["ticker"].map(lambda t: tick_side[t][1])
    m["mid"] = mid(m["yes_bid"], m["yes_ask"], m["last_price"])
    p = m.pivot_table(index=["espn_id", "snap"], columns="side", values=["mid", "volume", "open_interest"],
                      aggfunc="last")
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    p["kalshi_home_prob"] = p[[c for c in ("mid_home",) if c in p]].mean(axis=1)
    if "mid_away" in p:
        p["kalshi_home_prob"] = pd.concat([p.get("mid_home"), 1 - p["mid_away"]], axis=1).mean(axis=1)
    p["kalshi_volume"] = p.filter(like="volume_").sum(axis=1, min_count=1)
    p["kalshi_oi"] = p.filter(like="open_interest_").sum(axis=1, min_count=1)
    snaps = p[["espn_id", "snap", "kalshi_home_prob", "kalshi_volume", "kalshi_oi"]]

    # resting book: $ supporting each side (yes on X == no on the other team)
    b = load(data_dir, "kalshi_books")
    if not b.empty:
        num(b, ["price", "size"])
        b = b[b["ticker"].isin(tick_side)].copy()
        b["snap"] = ts(b["snapshot_ts"])
        b["espn_id"] = b["ticker"].map(lambda t: tick_side[t][0])
        team = b["ticker"].map(lambda t: tick_side[t][1])
        b["backs"] = np.where(b["side"] == "yes", team, np.where(team == "home", "away", "home"))
        b["usd"] = b["price"] * b["size"]
        imb = b.pivot_table(index=["espn_id", "snap"], columns="backs", values="usd", aggfunc="sum").reset_index()
        imb["kalshi_book_imbalance"] = (imb.get("home", 0) - imb.get("away", 0)) / (imb.get("home", 0) + imb.get("away", 0))
        snaps = snaps.merge(imb[["espn_id", "snap", "kalshi_book_imbalance"]], on=["espn_id", "snap"], how="left")

    flows = pd.DataFrame()
    t = load(data_dir, "kalshi_trades")
    if not t.empty:
        num(t, ["yes_price", "no_price", "count"])
        t = t.drop_duplicates("trade_id")
        t = t[t["ticker"].isin(tick_side)].copy()
        t["espn_id"] = t["ticker"].map(lambda x: tick_side[x][0])
        team = t["ticker"].map(lambda x: tick_side[x][1])
        paid = np.where(t["taker_side"] == "yes", t["yes_price"], t["no_price"]) * t["count"]
        backs_home = (t["taker_side"] == "yes") == (team == "home")
        t["flow"] = np.where(backs_home, paid, -paid)
        t["trade_dt"] = ts(t["created_time"])
        t["usd"] = paid
        flows = t.groupby(["espn_id", t["trade_dt"].dt.floor("h").rename("hour")]).agg(
            kalshi_flow=("flow", "sum"), kalshi_traded_usd=("usd", "sum")).reset_index()
    return snaps, flows


# ------------------------------------------------------------------ Polymarket

def build_poly(data_dir: Path, games: pd.DataFrame, matcher: Matcher, unmatched: list):
    m = load(data_dir, "poly_markets")
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = m[m.apply(lambda r: is_moneyline(r.where(r.notna(), None).to_dict()), axis=1)].copy()
    num(m, ["best_bid", "best_ask", "last_trade_price", "price_0", "price_1", "volume", "liquidity"])
    m["snap"] = ts(m["snapshot_ts"])
    cond: dict[str, tuple[str, int]] = {}      # condition_id -> (espn_id, index of home outcome)
    tok: dict[str, tuple[str, str]] = {}       # token -> (espn_id, side)
    for cid, d in m.groupby("condition_id"):
        r = d.iloc[-1]
        if pd.isna(r["outcome_0"]) or pd.isna(r["outcome_1"]):
            continue
        gid, sides = matcher.match(r["outcome_0"], r["outcome_1"], ts(pd.Series([r["game_start_time"]]))[0], 12)
        if gid is None:
            unmatched.append({"source": "polymarket", "id": cid, "team_a": r["outcome_0"],
                              "team_b": r["outcome_1"], "time": r["game_start_time"]})
            continue
        hi = 0 if sides[r["outcome_0"]] == "home" else 1
        cond[cid] = (gid, hi)
        for i in (0, 1):
            tok[str(r[f"token_{i}"])] = (gid, "home" if i == hi else "away")
    m = m[m["condition_id"].isin(cond)].copy()
    m["espn_id"] = m["condition_id"].map(lambda c: cond[c][0])
    hi = m["condition_id"].map(lambda c: cond[c][1])
    mid0 = mid(m["best_bid"], m["best_ask"], m["price_0"])       # best bid/ask quote outcome_0
    m["poly_home_prob"] = np.where(hi == 0, mid0, 1 - mid0)
    m["poly_volume"] = m["volume"]
    snaps = m.groupby(["espn_id", "snap"])[["poly_home_prob", "poly_volume"]].last().reset_index()

    b = load(data_dir, "poly_books")
    if not b.empty:
        num(b, ["price", "size"])
        b = b[(b["side"] == "bid") & b["token_id"].isin(tok)].copy()
        b["snap"] = ts(b["snapshot_ts"])
        b["espn_id"] = b["token_id"].map(lambda x: tok[x][0])
        b["backs"] = b["token_id"].map(lambda x: tok[x][1])
        b["usd"] = b["price"] * b["size"]
        imb = b.pivot_table(index=["espn_id", "snap"], columns="backs", values="usd", aggfunc="sum").reset_index()
        imb["poly_book_imbalance"] = (imb.get("home", 0) - imb.get("away", 0)) / (imb.get("home", 0) + imb.get("away", 0))
        snaps = snaps.merge(imb[["espn_id", "snap", "poly_book_imbalance"]], on=["espn_id", "snap"], how="left")

    flows = pd.DataFrame()
    t = load(data_dir, "poly_trades")
    if not t.empty:
        num(t, ["price", "size"])
        t = t.drop_duplicates("trade_id")
        t = t[t["token_id"].isin(tok)].copy()
        t["espn_id"] = t["token_id"].map(lambda x: tok[x][0])
        backs_home = (t["token_id"].map(lambda x: tok[x][1]) == "home") == (t["side"].str.upper() == "BUY")
        t["usd"] = t["price"] * t["size"]
        t["flow"] = np.where(backs_home, t["usd"], -t["usd"])
        t["trade_dt"] = ts(t["trade_ts"])
        flows = t.groupby(["espn_id", t["trade_dt"].dt.floor("h").rename("hour")]).agg(
            poly_flow=("flow", "sum"), poly_traded_usd=("usd", "sum")).reset_index()
    return snaps, flows


# ------------------------------------------------------------------ panel + report

def hourly(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.merge(games[["espn_id", "kickoff_dt"]], on="espn_id")
    df = df[df["snap"] < df["kickoff_dt"]].copy()               # stop at kickoff
    df["hour"] = df["snap"].dt.floor("h")
    return df.sort_values("snap").groupby(["espn_id", "hour"]).last().reset_index().drop(
        columns=["snap", "kickoff_dt"])


def open_close(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["espn_id"])
    d = df.sort_values("hour")
    first = d.groupby("espn_id")[cols].first().add_suffix("_open")
    last = d.groupby("espn_id")[cols].last().add_suffix("_close")
    out = first.join(last)
    out["first_seen"] = d.groupby("espn_id")["hour"].min()
    return out.reset_index()


def corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 5 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return np.nan, np.nan, len(d)
    return (d.iloc[:, 0].corr(d.iloc[:, 1]),
            d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()), len(d))


def build_report(panel: pd.DataFrame, summary: pd.DataFrame) -> str:
    L = ["# NCAAF line movement vs prediction markets", ""]
    L.append(f"Games: {len(summary)}; finals: {int(summary['final'].sum()) if 'final' in summary else 0}; "
             f"game-hours (pre-kickoff): {len(panel)}")
    L.append("")
    if panel.empty:
        return "\n".join(L + ["No panel data yet."])
    p = panel.sort_values(["espn_id", "hour"]).copy()
    gap = p.groupby("espn_id")["hour"].diff() == pd.Timedelta(hours=1)
    if {"an_money_home_pct", "an_tickets_home_pct"} <= set(p):
        # money% well above tickets% on a side = bigger (sharper) bettors on that side
        p["an_sharp_home"] = p["an_money_home_pct"] - p["an_tickets_home_pct"]
        p["an_money_home_lean"] = p["an_money_home_pct"] - 50
    for c in ("book_home_prob", "book_spread", "book_total", "kalshi_home_prob", "poly_home_prob",
              "kalshi_book_imbalance", "poly_book_imbalance", "an_money_home_pct", "an_sharp_home"):
        if c in p:
            p["d_" + c] = p.groupby("espn_id")[c].diff().where(gap)
    for c in ("kalshi_flow", "poly_flow"):
        if c in p:
            p[c] = p[c].fillna(0)

    L += ["## Same-hour correlation of changes", "",
          "Hourly change in consensus sportsbook P(home) vs prediction-market signals.",
          "Spread correlations should be negative (home spread falls when home gets stronger).", "",
          "| book signal | market signal | pearson | spearman | n |", "|---|---|---|---|---|"]
    pairs = [("d_book_home_prob", "d_kalshi_home_prob"), ("d_book_home_prob", "d_poly_home_prob"),
             ("d_book_home_prob", "kalshi_flow"), ("d_book_home_prob", "poly_flow"),
             ("d_book_home_prob", "d_kalshi_book_imbalance"), ("d_book_home_prob", "d_poly_book_imbalance"),
             ("d_book_spread", "d_kalshi_home_prob"), ("d_book_spread", "d_poly_home_prob"),
             ("d_book_spread", "kalshi_flow"), ("d_book_spread", "poly_flow"),
             ("d_book_home_prob", "an_money_home_lean"), ("d_book_home_prob", "d_an_money_home_pct"),
             ("d_book_home_prob", "an_sharp_home"), ("d_book_spread", "an_sharp_home")]
    for a, b in pairs:
        if a in p and b in p:
            pr, sp, n = corr(p[a], p[b])
            L.append(f"| {a} | {b} | {pr:.3f} | {sp:.3f} | {n} |")

    L += ["", "## Lead / lag", "",
          "corr(book change at hour t, market signal at hour t+k). Peak at k<0 => the prediction",
          "market moved FIRST (books followed); k>0 => books led.", "",
          "| market signal | " + " | ".join(f"k={k}" for k in range(-3, 4)) + " |",
          "|---|" + "---|" * 7]
    for b in ("d_kalshi_home_prob", "d_poly_home_prob", "kalshi_flow", "poly_flow",
              "d_an_money_home_pct", "an_sharp_home"):
        if b not in p or "d_book_home_prob" not in p:
            continue
        cells = []
        for k in range(-3, 4):
            shifted = p.groupby("espn_id")[b].shift(-k)
            cells.append(f"{corr(p['d_book_home_prob'], shifted)[0]:.3f}")
        L.append(f"| {b} | " + " | ".join(cells) + " |")

    if "d_book_spread" in p:
        big = p[(p["d_book_spread"].abs() >= 1) | (p.get("d_book_home_prob", pd.Series(0, index=p.index)).abs() >= 0.02)]
        L += ["", "## Do big book moves have prediction-market money behind them?", "",
              f"Book moves >= 1 pt spread or >= 2% win prob in one hour: {len(big)}", ""]
        direction = -np.sign(big["d_book_spread"]).where(big["d_book_spread"] != 0,
                                                          np.sign(big.get("d_book_home_prob", 0)))
        for c, label in (("kalshi_flow", "Kalshi net taker flow"), ("poly_flow", "Polymarket net taker flow"),
                         ("d_kalshi_home_prob", "Kalshi price change"), ("d_poly_home_prob", "Polymarket price change"),
                         ("an_sharp_home", "Public splits: money% minus tickets% (big-bettor side)"),
                         ("an_money_home_lean", "Public splits: side with majority of money")):
            if c not in big:
                continue
            prev = p.groupby("espn_id")[c].shift(1).reindex(big.index).fillna(0) if "flow" in c else 0
            sig = np.sign(big[c].fillna(0) + prev)
            has = sig != 0
            agree = (sig[has] == direction[has]).mean() if has.any() else np.nan
            window = "same + prior hour" if "flow" in c else "same hour"
            L.append(f"- {label} ({window}) agrees with move direction: "
                     f"{agree:.1%} of {int(has.sum())} moves with activity" if has.any() else
                     f"- {label}: no activity during big moves")

    s = summary
    L += ["", "## Game level: open -> close", ""]
    if {"book_home_prob_open", "book_home_prob_close"} <= set(s):
        s = s.copy()
        s["book_move"] = s["book_home_prob_close"] - s["book_home_prob_open"]
        for c in ("kalshi_home_prob", "poly_home_prob"):
            if f"{c}_open" in s:
                s[f"{c}_move"] = s[f"{c}_close"] - s[f"{c}_open"]
                pr, sp, n = corr(s["book_move"], s[f"{c}_move"])
                L.append(f"- book P(home) move vs {c} move: pearson {pr:.3f}, spearman {sp:.3f}, n={n}")
        for c in ("kalshi_flow_total", "poly_flow_total"):
            if c in s:
                pr, sp, n = corr(s["book_move"], s[c])
                L.append(f"- book P(home) move vs {c}: pearson {pr:.3f}, spearman {sp:.3f}, n={n}")

    fin = s[s.get("final", False) == True] if "final" in s else s.iloc[0:0]  # noqa: E712
    L += ["", "## Results (wins/losses) vs lines", ""]
    if len(fin):
        for c, label in (("book_home_prob_close", "Sportsbook close (consensus)"),
                         ("sharp_home_prob_close", "Pinnacle close"),
                         ("kalshi_home_prob_close", "Kalshi pre-kickoff"),
                         ("poly_home_prob_close", "Polymarket pre-kickoff"),
                         ("book_home_prob_open", "Sportsbook first tracked")):
            if c in fin:
                d = fin[[c, "home_won"]].dropna()
                if len(d):
                    brier = ((d[c] - d["home_won"]) ** 2).mean()
                    acc = ((d[c] > 0.5) == (d["home_won"] == 1)).mean()
                    L.append(f"- {label}: Brier {brier:.4f}, favorite won {acc:.1%} (n={len(d)})")
        if {"book_spread_open", "book_spread_close", "margin_home"} <= set(fin):
            d = fin.dropna(subset=["book_spread_open", "book_spread_close", "margin_home"])
            moved = d[d["book_spread_close"] != d["book_spread_open"]]
            if len(moved):
                toward_home = moved["book_spread_close"] < moved["book_spread_open"]
                ats_open = moved["margin_home"] + moved["book_spread_open"]
                win = np.where(toward_home, ats_open > 0, ats_open < 0)
                push = ats_open == 0
                L.append(f"- Following the open->close move against the OPENING spread: "
                         f"{win[~push].mean():.1%} ATS (n={int((~push).sum())}; 52.4% breaks even at -110)")
                ats_close = d["margin_home"] + d["book_spread_close"]
                L.append(f"- Home cover rate vs closing spread: {(ats_close > 0)[ats_close != 0].mean():.1%} "
                         f"(should be ~50% if closing lines are efficient)")
    else:
        L.append("No finished games yet.")
    L += espn_line_section(fin)
    L.append("")
    return "\n".join(L)


def espn_line_section(fin: pd.DataFrame) -> list[str]:
    """ESPN's provider (DraftKings) open/close: available for every game incl. backfill."""
    need = {"espn_spread_home_open", "espn_spread_home_close", "margin_home"}
    if fin.empty or not need <= set(fin):
        return []
    L = ["", "### ESPN provider (DraftKings) open -> close, all finished games", ""]
    d = fin.dropna(subset=list(need))
    if {"espn_ml_home_close", "espn_ml_away_close"} <= set(d):
        m = d.dropna(subset=["espn_ml_home_close", "espn_ml_away_close"])
        ph, pa = american_to_prob(m["espn_ml_home_close"]), american_to_prob(m["espn_ml_away_close"])
        p = ph / (ph + pa)
        if len(m):
            L.append(f"- Closing ML favorite won {((p > 0.5) == (m['home_won'] == 1)).mean():.1%}; "
                     f"Brier {((p - m['home_won']) ** 2).mean():.4f} (n={len(m)})")
    ats_close = d["margin_home"] + d["espn_spread_home_close"]
    L.append(f"- Home cover rate vs closing spread: {(ats_close > 0)[ats_close != 0].mean():.1%} (n={int((ats_close != 0).sum())})")
    moved = d[d["espn_spread_home_close"] != d["espn_spread_home_open"]]
    if len(moved):
        toward_home = moved["espn_spread_home_close"] < moved["espn_spread_home_open"]
        ats_open = moved["margin_home"] + moved["espn_spread_home_open"]
        push = ats_open == 0
        win = np.where(toward_home, ats_open > 0, ats_open < 0)[~push]
        L.append(f"- Spread moved in {len(moved)}/{len(d)} games; following the move vs the OPENING "
                 f"spread went {win.mean():.1%} ATS (n={len(win)}; 52.4% breaks even at -110)")
        big = moved[(moved["espn_spread_home_close"] - moved["espn_spread_home_open"]).abs() >= 2]
        if len(big):
            tb = big["espn_spread_home_close"] < big["espn_spread_home_open"]
            ao = big["margin_home"] + big["espn_spread_home_open"]
            w = np.where(tb, ao > 0, ao < 0)[ao != 0]
            L.append(f"- Moves of 2+ points: following them went {w.mean():.1%} ATS vs opener (n={len(w)})")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--aliases", default="aliases.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    games = build_games(data_dir)
    if games.empty:
        log.error("no espn_games data in %s", data_dir)
        return 1
    matcher = Matcher(games, load_aliases(Path(args.aliases)))
    unmatched: list[dict] = []
    runs = load(data_dir, "runs")
    first_run = ts(runs["run_ts"]).min() if not runs.empty else pd.Timestamp.min.tz_localize("UTC")

    lines, mapping, history_ids = build_book_lines(data_dir, games, matcher, unmatched)
    boc = book_open_close(lines, first_run, history_ids)
    cons = hourly(consensus(lines), games)
    splits = hourly(build_splits(data_dir, mapping), games)
    k_snaps, k_flows = build_kalshi(data_dir, games, matcher, unmatched)
    p_snaps, p_flows = build_poly(data_dir, games, matcher, unmatched)
    k_h, p_h = hourly(k_snaps, games), hourly(p_snaps, games)

    panel = pd.DataFrame(columns=["espn_id", "hour"])
    for part in (cons, splits, k_h, p_h):
        if not part.empty:
            panel = panel.merge(part, on=["espn_id", "hour"], how="outer")
    ko = games.set_index("espn_id")["kickoff_dt"]
    for fl in (k_flows, p_flows):
        if not fl.empty:
            fl = fl[fl["hour"] < fl["espn_id"].map(ko)]
            panel = panel.merge(fl, on=["espn_id", "hour"], how="outer")
    if not panel.empty:
        panel["hours_to_kickoff"] = (panel["espn_id"].map(ko) - panel["hour"]).dt.total_seconds() / 3600
        panel = panel.merge(games[["espn_id", "home_name", "away_name"]], on="espn_id", how="left")
        panel = panel.sort_values(["espn_id", "hour"])

    summary = games.copy()
    for part, cols in ((cons, ["book_home_prob", "book_spread", "book_total", "sharp_home_prob", "sharp_spread"]),
                       (splits, ["an_money_home_pct", "an_tickets_home_pct", "an_ml_money_home_pct"]),
                       (k_h, ["kalshi_home_prob", "kalshi_volume", "kalshi_oi"]),
                       (p_h, ["poly_home_prob", "poly_volume"])):
        cols = [c for c in cols if c in part]
        if cols:
            oc = open_close(part, cols)
            oc = oc.rename(columns={"first_seen": f"{cols[0].split('_')[0]}_first_seen"})
            summary = summary.merge(oc, on="espn_id", how="left")
    for fl, name in ((k_flows, "kalshi"), (p_flows, "poly")):
        if not fl.empty:
            f = fl[fl["hour"] < fl["espn_id"].map(ko)].groupby("espn_id").agg(
                **{f"{name}_flow_total": (f"{name}_flow", "sum"),
                   f"{name}_traded_usd_total": (f"{name}_traded_usd", "sum")}).reset_index()
            summary = summary.merge(f, on="espn_id", how="left")
    if "book_first_seen" in summary:
        summary["opener_censored"] = ((summary["book_first_seen"] <= first_run + pd.Timedelta(hours=1))
                                      & ~summary["espn_id"].isin(history_ids))
    if not boc.empty:
        mo = boc[boc["book"] == "an_open"].pivot_table(index="espn_id", columns="market", values="open",
                                                       aggfunc="first")
        mo = mo.rename(columns={"spread_home": "market_open_spread", "total": "market_open_total",
                                "home_prob": "market_open_home_prob", "ml_home": "market_open_ml_home"})
        summary = summary.merge(mo.reset_index(), on="espn_id", how="left")
    if {"book_spread_close", "margin_home"} <= set(summary):
        ats = summary["margin_home"] + summary["book_spread_close"]
        summary["home_cover_close"] = np.select([ats > 0, ats < 0, ats == 0], ["cover", "no_cover", "push"], None)
    if {"book_total_close", "total_points"} <= set(summary):
        diff = summary["total_points"] - summary["book_total_close"]
        summary["total_result_close"] = np.select([diff > 0, diff < 0, diff == 0], ["over", "under", "push"], None)
    summary = summary.drop(columns=[c for c in summary if c.endswith("_raw")])
    summary = summary.sort_values("kickoff_dt")

    summary.to_csv(out / "games_summary.csv", index=False)
    boc.to_csv(out / "book_open_close.csv", index=False)
    lines.to_csv(out / "line_history.csv", index=False)
    panel.to_csv(out / "hourly_panel.csv", index=False)
    pd.DataFrame(unmatched).to_csv(out / "unmatched.csv", index=False)
    (out / "report.md").write_text(build_report(panel, summary))
    log.info("games=%d book_lines=%d panel_rows=%d unmatched=%d -> %s",
             len(summary), len(lines), len(panel), len(unmatched), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
