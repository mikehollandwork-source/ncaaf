"""Offline end-to-end test: fake API payloads -> parsers -> storage -> analysis."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from ncaaf_tracker import analyze
from ncaaf_tracker.matching import Matcher, name_score, norm
from ncaaf_tracker.sources import espn, kalshi, oddsapi, polymarket
from ncaaf_tracker.util import iso, parse_time, write_table

KICK = datetime(2026, 10, 3, 19, 30, tzinfo=timezone.utc)
START = KICK - timedelta(hours=6)


def espn_event(state="pre", hs=None, as_=None):
    return {
        "id": "401", "name": "Ohio State Buckeyes at Michigan Wolverines",
        "season": {"year": 2026, "type": 2}, "week": {"number": 6},
        "competitions": [{
            "date": "2026-10-03T19:30Z", "neutralSite": False,
            "status": {"type": {"state": state, "completed": state == "post", "detail": "x"}},
            "competitors": [
                {"homeAway": "home", "score": hs, "winner": state == "post" and hs > as_, "curatedRank": {"current": 99},
                 "team": {"id": "130", "displayName": "Michigan Wolverines", "shortDisplayName": "Michigan",
                          "location": "Michigan", "name": "Wolverines", "abbreviation": "MICH"}},
                {"homeAway": "away", "score": as_, "winner": state == "post" and as_ > hs, "curatedRank": {"current": 4},
                 "team": {"id": "194", "displayName": "Ohio State Buckeyes", "shortDisplayName": "Ohio State",
                          "location": "Ohio State", "name": "Buckeyes", "abbreviation": "OSU"}},
            ],
            "odds": [{"provider": {"name": "ESPN BET"}, "details": "OSU -3.5", "spread": 3.5, "overUnder": 47.5,
                      "pointSpread": {"home": {"open": {"line": "+2.5"}, "close": {"line": "+3.5"}}},
                      "total": {"over": {"open": {"line": "o49.5"}, "close": {"line": "o47.5"}}},
                      "moneyline": {"home": {"open": {"odds": "+120"}, "close": {"odds": "+145"}}}}],
        }],
    }


def odds_payload(spread_home, ml_home, ml_away, total):
    books = []
    for key, adj in (("draftkings", 0), ("fanduel", 0.5), ("pinnacle", 0)):
        books.append({"key": key, "last_update": "x", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Michigan Wolverines", "price": ml_home},
                                        {"name": "Ohio State Buckeyes", "price": ml_away}]},
            {"key": "spreads", "outcomes": [{"name": "Michigan Wolverines", "price": -110, "point": spread_home + adj},
                                            {"name": "Ohio State Buckeyes", "price": -110, "point": -spread_home - adj}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": total},
                                           {"name": "Under", "price": -110, "point": total}]},
        ]})
    return [{"id": "oa1", "commence_time": "2026-10-03T19:30:00Z", "home_team": "Michigan Wolverines",
             "away_team": "Ohio State Buckeyes", "bookmakers": books}]


def kalshi_market(ticker, team, bid, ask, vol):
    return {"ticker": ticker, "event_ticker": "KXNCAAFGAME-26OCT03OSUMICH", "title": "Ohio St. vs Michigan",
            "yes_sub_title": team, "status": "active",
            "expected_expiration_time": iso(KICK + timedelta(hours=3, minutes=30)),
            "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}", "volume_fp": str(vol)}


def poly_event(p_osu, vol):
    return [{"id": "e1", "title": "Ohio State vs. Michigan", "slug": "cfb-osu-mich", "markets": [{
        "id": "m1", "conditionId": "0xc1", "question": "Ohio State vs. Michigan", "sportsMarketType": "moneyline",
        "gameStartTime": "2026-10-03 19:30:00+00", "outcomes": '["Ohio State", "Michigan"]',
        "outcomePrices": f'["{p_osu}", "{1 - p_osu:.3f}"]', "clobTokenIds": '["tokOSU", "tokMICH"]',
        "bestBid": p_osu - 0.01, "bestAsk": p_osu + 0.01, "volumeNum": vol}]},
        {"id": "e2", "title": "CFP Champion", "markets": [{"id": "fut", "question": "Who wins?"}]}]


def build_data(tmp_path):
    d = tmp_path / "data"
    # Line drifts toward Ohio State over 5 hourly snapshots; PM money follows.
    spreads = [2.5, 3.0, 3.5, 5.0, 5.5, 5.0]
    probs = [0.55, 0.56, 0.57, 0.60, 0.61, 0.60]
    for h in range(6):
        now = START + timedelta(hours=h)
        sp, p = spreads[h], probs[h]
        write_table(d, "runs", [{"run_ts": iso(now), "mode": "full"}], now)
        g = espn.parse_event(espn_event())
        g["snapshot_ts"] = iso(now)
        write_table(d, "espn_games", [g], now)
        write_table(d, "odds", oddsapi.parse_odds(odds_payload(sp, int(120 + 10 * (sp - 2.5)), int(-140 - 10 * (sp - 2.5)), 49.5 - 0.5 * h), now), now)
        km = [kalshi.market_row(kalshi_market("K-OSU", "Ohio St.", p - 0.01, p + 0.01, 1000 * (h + 1)), "KXNCAAFGAME", iso(now)),
              kalshi.market_row(kalshi_market("K-MICH", "Michigan", 1 - p - 0.01, 1 - p + 0.01, 900 * (h + 1)), "KXNCAAFGAME", iso(now))]
        write_table(d, "kalshi_markets", km, now)
        ob = {"orderbook_fp": {"yes_dollars": [[f"{p - 0.01:.2f}", "500"]], "no_dollars": [[f"{1 - p - 0.01:.2f}", "100"]]}}
        write_table(d, "kalshi_books", kalshi.parse_orderbook(ob, "K-OSU", iso(now), 10), now)
        write_table(d, "kalshi_trades", [{"trade_id": f"t{h}", "ticker": "K-OSU", "created_time": iso(now - timedelta(minutes=20)),
                                          "yes_price": p, "no_price": 1 - p, "count": 100 * (h + 1), "taker_side": "yes"}], now)
        write_table(d, "poly_markets", polymarket.market_rows(poly_event(p, 5000 * (h + 1)), now, None), now)
        write_table(d, "poly_trades", [{"trade_id": f"p{h}", "condition_id": "0xc1", "token_id": "tokOSU", "side": "BUY",
                                        "price": p, "size": 200, "trade_ts": iso(now - timedelta(minutes=10))}], now)
    # in-game snapshot must be ignored; final result recorded
    after = KICK + timedelta(hours=4)
    live = oddsapi.parse_odds(odds_payload(-20, -2000, 1000, 30), KICK - timedelta(seconds=1))
    for r in live:
        r["snapshot_ts"] = iso(KICK + timedelta(minutes=5))
    write_table(d, "odds", live, after)
    g = espn.parse_event(espn_event("post", 17, 27))
    g["snapshot_ts"] = iso(after)
    write_table(d, "espn_games", [g], after)
    return d


def test_parse_time_formats():
    assert parse_time("2026-10-03T19:30Z") == KICK
    assert parse_time("2026-10-03 19:30:00+00") == KICK
    assert parse_time(int(KICK.timestamp())) == KICK
    assert parse_time(int(KICK.timestamp() * 1000)) == KICK


def test_name_matching():
    assert norm("Ohio St.") == "ohio state"
    assert name_score("Ohio St.", ["ohio state buckeyes", "ohio state"]) == 1.0
    assert name_score("Ole Miss", [norm("Ole Miss Rebels"), norm("Ole Miss")]) == 1.0
    games = pd.DataFrame([espn.parse_event(espn_event())])
    m = Matcher(games)
    gid, sides = m.match("Ohio St.", "Michigan", KICK, 6)
    assert gid == "401" and sides == {"Ohio St.": "away", "Michigan": "home"}
    assert m.match("Alabama", "Georgia", KICK, 6) == (None, None)
    assert norm("Tennessee-Martin") == norm("UT Martin")
    assert norm("SE Louisiana") == norm("Southeastern Louisiana")
    # ESPN lists kickoff-TBD games at 04:00Z; still match a same-day market time
    tbd = games.copy()
    tbd["kickoff"] = "2026-10-04T04:00:00Z"
    assert Matcher(tbd).match("Ohio St.", "Michigan", KICK, 6)[0] == "401"


def test_results_table():
    final = espn.parse_event(espn_event("post", 17, 27))
    r = espn.results([final, espn.parse_event(espn_event())])
    assert len(r) == 1
    r = r[0]
    assert (r["home_team"], r["away_team"]) == ("Michigan Wolverines", "Ohio State Buckeyes")
    assert (r["home_score"], r["away_score"], r["winner_side"]) == (17, 27, "away")
    assert (r["winner"], r["loser"], r["home_margin"]) == ("Ohio State Buckeyes", "Michigan Wolverines", -10)
    postponed = espn.parse_event(espn_event("post", 0, 0))
    postponed["completed"] = False
    assert espn.results([postponed]) == []


def test_espn_odds_parse():
    g = espn.parse_event(espn_event())
    assert g["away_rank"] == 4 and g["home_rank"] is None   # 99 = unranked
    assert g["espn_spread_home_open"] == 2.5 and g["espn_total_open"] == 49.5
    assert g["espn_ml_home_close"] == 145


def test_oddsapi_stops_at_kickoff():
    assert oddsapi.parse_odds(odds_payload(3, 120, -140, 48), KICK) == []
    assert len(oddsapi.parse_odds(odds_payload(3, 120, -140, 48), KICK - timedelta(minutes=1))) == 18


def test_kalshi_parsers():
    r = kalshi.market_row({"ticker": "X", "event_ticker": "KXNCAAFGAME-26SEP05ECUALA", "yes_bid": 42, "yes_ask": 45},
                          "KXNCAAFGAME", "t")
    assert r["yes_bid"] == 0.42 and r["ticker_date"] == "2026-09-05T00:00:00Z"
    legacy = kalshi.parse_orderbook({"orderbook": {"yes": [[40, 10], [42, 5]], "no": None}}, "X", "t", 10)
    assert legacy[0]["price"] == 0.42 and legacy[0]["level"] == 0


def test_polymarket_skips_futures():
    rows = polymarket.market_rows(poly_event(0.6, 1), START, None)
    assert len(rows) == 1 and rows[0]["outcome_0"] == "Ohio State" and polymarket.is_moneyline(rows[0])


def test_end_to_end(tmp_path):
    d = build_data(tmp_path)
    out = tmp_path / "out"
    assert analyze.main(["--data-dir", str(d), "--out", str(out), "--aliases", str(tmp_path / "none.csv")]) == 0
    s = pd.read_csv(out / "games_summary.csv")
    assert len(s) == 1
    g = s.iloc[0]
    assert g["winner"] == "Ohio State Buckeyes" and g["margin_home"] == -10
    assert g["book_spread_open"] == 2.5 and g["book_spread_close"] == 5.0  # in-game snapshot excluded
    assert g["book_total_open"] == 49.5 and g["book_total_close"] == 47.0
    assert g["espn_spread_home_open"] == 2.5
    assert g["home_cover_close"] == "no_cover" and g["total_result_close"] == "under"
    assert g["kalshi_home_prob_open"] < 0.46 and g["kalshi_home_prob_close"] < g["kalshi_home_prob_open"]
    assert g["poly_home_prob_close"] < g["poly_home_prob_open"]
    assert g["kalshi_flow_total"] < 0 and g["poly_flow_total"] < 0  # money on the away team
    assert bool(g["opener_censored"])
    boc = pd.read_csv(out / "book_open_close.csv")
    dk = boc[(boc["book"] == "draftkings") & (boc["market"] == "spread_home")].iloc[0]
    assert dk["open"] == 2.5 and dk["close"] == 5.0 and dk["n_changes"] == 5
    panel = pd.read_csv(out / "hourly_panel.csv")
    assert len(panel) == 6 and (panel["hours_to_kickoff"] > 0).all()
    assert pd.read_csv(out / "unmatched.csv").empty if (out / "unmatched.csv").stat().st_size > 1 else True
    report = (out / "report.md").read_text()
    assert "Lead / lag" in report and "Brier" in report


# ------------------------------------------------------------------ keyless site feeds (real payloads)
import json
from pathlib import Path

from ncaaf_tracker import analyze as _an
from ncaaf_tracker.sources import scraped

FIX = Path(__file__).parent / "fixtures"


def test_actionnetwork_scoreboard_real_payload():
    games = json.loads((FIX / "an_scoreboard.json").read_text())["games"]
    g = next(x for x in games if x["id"] == 289186)            # Texas @ Tennessee
    g["status"] = "scheduled"
    odds, splits = scraped.an_parse_game(g, parse_time("2026-09-26T12:00:00Z"))
    dk = {(r["market"], r["outcome"]): r for r in odds if r["book"] == "draftkings"}
    assert dk[("spreads", "Tennessee Volunteers")]["point"] == 5.5
    assert dk[("h2h", "Texas Longhorns")]["price"] == -205
    assert dk[("totals", "Over")]["point"] == 54.5
    opener = {(r["market"], r["outcome"]): r for r in odds if r["book"] == "an_open"}
    assert opener[("spreads", "Tennessee Volunteers")]["point"] == 6.5
    sp = {(r["market"], r["side"]): r for r in splits}
    assert sp[("spreads", "away")]["money_pct"] == 51 and sp[("h2h", "away")]["money_pct"] == 79
    # stops at kickoff / skips finished games
    assert scraped.an_parse_game(g, parse_time("2026-09-26T16:00:00Z")) == ([], [])


def test_actionnetwork_history_real_payload(tmp_path):
    games = json.loads((FIX / "an_scoreboard.json").read_text())["games"]
    g = next(x for x in games if x["id"] == 289145)            # Coastal Carolina @ Liberty
    hist = json.loads((FIX / "an_history_289145.json").read_text())

    class S:  # fake session returning the saved payload
        def get(self, *a, **k):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self): return hist
            return R()
    rows = scraped.an_history(S(), g, parse_time("2026-09-27T00:00:00Z"))
    dk = [r for r in rows if r["book"] == "draftkings" and r["market"] == "spreads"]
    assert dk and dk[0]["line_status"] == "opener"
    snaps = _an.history_snapshots(pd.DataFrame(rows))
    ko = parse_time("2026-09-24T23:30:00Z")
    t = pd.to_datetime(snaps["snapshot_ts"], utc=True)
    assert (t < ko).all() and len(t.unique()) > 50
    home_spread = snaps[(snaps.book == "draftkings") & (snaps.market == "spreads")
                        & (snaps.outcome == g["teams"][1]["full_name"] if g["home_team_id"] == g["teams"][1]["id"]
                           else snaps.outcome == g["teams"][0]["full_name"])]
    first = home_spread.sort_values("snapshot_ts").iloc[0]
    assert float(first["point"]) == -1.5                          # DK opener, home side


def test_pinnacle_parse_real_shape():
    matchups = json.loads((FIX / "pin_matchups.json").read_text())
    markets = json.loads((FIX / "pin_markets.json").read_text())
    live = next(m for m in matchups if m["id"] == 1637188539)    # Rice @ Fresno State (live copy)
    pre = dict(live, isLive=False, parentId=None, id=999)
    mk = [dict(k, matchupId=999, period=0) for k in markets if k["matchupId"] == 1637188539]
    rows = scraped.pin_parse(matchups + [pre], markets + mk, parse_time("2026-09-26T20:00:00Z"))
    assert rows and {r["book"] for r in rows} == {"pinnacle"}
    spread = [r for r in rows if r["market"] == "spreads" and r["outcome"] == "Fresno State"]
    assert [r["point"] for r in spread] == [-4.5]                  # main line only, no alternates
    assert not [r for r in rows if "Heisman" in str(r)]            # futures skipped


def test_last_history_save(tmp_path):
    from ncaaf_tracker.collect import last_history_save
    assert last_history_save(tmp_path) is None
    write_table(tmp_path, "an_history", [{"a": 1}], parse_time("2026-09-27T10:28:34Z"))
    write_table(tmp_path, "an_history", [{"a": 1}], parse_time("2026-10-02T03:00:00Z"))
    assert last_history_save(tmp_path) == parse_time("2026-10-02T03:00:00Z")


def test_box_score_real_payload(tmp_path):
    data = json.loads((FIX / "espn_summary_401858242.json").read_text())
    team, players = espn.parse_box_score(data, "401858242")
    vt = {r["stat"]: r for r in team if r["team"] == "Virginia Tech Hokies"}
    assert vt["totalYards"]["value"] == 370 and vt["totalYards"]["home_away"] == "away"
    assert vt["completionAttempts"]["display"] == "22/35" and vt["thirdDownEff"]["display"] == "4-15"
    qb = {(r["category"], r["stat"]): r for r in players if r["player"] == "Ethan Grunkemeyer"}
    assert qb[("passing", "passingYards")]["value"] == 147
    assert qb[("passing", "completions")]["value"] == 22 and qb[("passing", "passingAttempts")]["value"] == 35
    assert qb[("passing", "adjQBR")]["value"] == 54.1 and qb[("passing", "adjQBR")]["home_away"] == "away"
    assert {r["category"] for r in players} >= {"passing", "rushing", "receiving", "defensive", "kicking"}

    # saved once per game: the next run skips games already stored
    from ncaaf_tracker.collect import saved_box_score_ids
    write_table(tmp_path, "box_team", team, parse_time("2026-10-03T00:00:00Z"))
    assert saved_box_score_ids(tmp_path) == {"401858242"}


def test_actionnetwork_props_real_payload():
    data = json.loads((FIX / "an_props_289385.json").read_text())
    game = {"id": 289385, "start_time": "2026-10-09T23:00:00.000Z", "home_team_id": 257, "away_team_id": 262,
            "teams": [{"id": 257, "full_name": "Louisville Cardinals"}, {"id": 262, "full_name": "Florida State Seminoles"}]}
    rows = scraped.an_parse_props(data, game, parse_time("2026-10-06T04:00:00Z"))
    py = [r for r in rows if r["prop_type"] == "core_bet_type_9_passing_yards" and r["player"] == "Ashton Daniels"]
    assert {(r["book"], r["side"], r["line"], r["odds"]) for r in py} >= {("fanduel", "over", 214.5, -114),
                                                                           ("an_consensus", "under", 214.5, -114)}
    assert {r["team"] for r in py} == {"Florida State Seminoles"} and {r["home_away"] for r in py} == {"away"}
    assert py[0]["player_info"] == "FSU - QB" and py[0]["milestone"] is False
    assert any(r["milestone"] for r in rows) and any(r["prop_group"] == "game_props" for r in rows)
    assert any(r["prop_name"] == "Anytime TD Scorer" for r in rows)
    tt = [r for r in rows if r["prop_name"] == "Team Total"]
    assert tt and all(r["home_away"] in ("home", "away") for r in tt)


def test_play_by_play_with_win_probability():
    data = json.loads((FIX / "espn_summary_401858242.json").read_text())
    plays = espn.parse_plays(data, "401858242")
    assert len(plays) == 186
    assert all(p["home_wp_after"] is not None for p in plays)
    # win probability chains play to play
    assert all(a["home_wp_after"] == b["home_wp_before"] for a, b in zip(plays, plays[1:]))
    assert any(p["yards"] and p["yards"] >= 20 for p in plays)
