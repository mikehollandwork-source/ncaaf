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


def test_kalshi_trades_window_and_paging():
    calls = []

    class S:
        def get(self, url, params=None, timeout=None):
            calls.append(dict(params))
            page = len(calls)

            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    if page == 1:
                        return {"trades": [{"trade_id": "a", "count_fp": "5", "yes_price_dollars": "0.40",
                                            "taker_side": "yes", "created_time": "2026-10-03T15:00:00Z"}], "cursor": "c1"}
                    return {"trades": [{"trade_id": "b", "count": 2, "yes_price": 41, "taker_side": "no",
                                        "created_time": "2026-10-03T14:00:00Z"}], "cursor": None}
            return R()
    since, until = parse_time("2026-09-26T00:00:00Z"), parse_time("2026-10-03T16:00:00Z")
    got = kalshi.fetch_trades(S(), "T", since, until)
    assert [t["trade_id"] for t in got] == ["a", "b"] and got[0]["count"] == 5 and got[1]["yes_price"] == 0.41
    assert calls[0]["min_ts"] == int(since.timestamp()) and calls[0]["max_ts"] == int(until.timestamp())
    assert calls[1]["cursor"] == "c1"


def test_locked_rule_snapshot():
    from ncaaf_tracker.collect import rule_snapshot
    games = [{"espn_id": "1", "kickoff": "2026-10-17T19:30:00Z", "home_name": "Tennessee Volunteers",
              "away_name": "Alabama Crimson Tide", "home_short": "Tennessee", "away_short": "Alabama",
              "home_location": "Tennessee", "away_location": "Alabama", "home_abbr": "TENN", "away_abbr": "ALA",
              "home_id": "2633", "away_id": "333", "home_rank": 14, "away_rank": 6}]
    base = {"event_id": "an:1", "commence_time": "2026-10-17T19:30:00Z", "home_team": "Tennessee Volunteers",
            "away_team": "Alabama Crimson Tide", "market": "spreads"}
    odds = [{**base, "book": "an_open", "outcome": "Tennessee Volunteers", "point": 5.5, "price": -110},
            {**base, "book": "draftkings", "outcome": "Tennessee Volunteers", "point": 4.0, "price": -108},
            {**base, "book": "draftkings", "outcome": "Alabama Crimson Tide", "point": -4.0, "price": -112}]
    splits = [{**base, "side": "home", "tickets_pct": 35, "money_pct": 40},
              {**base, "side": "away", "tickets_pct": 65, "money_pct": 60}]
    r = rule_snapshot(games, odds, splits, parse_time("2026-10-16T12:00:00Z"))[0]
    assert r["fires"] and r["pick_side"] == "home" and r["pick_line"] == 4.0 and r["pick_odds"] == -108
    splits[0]["money_pct"] = 55          # public money now with the move -> no bet
    assert not rule_snapshot(games, odds, splits, parse_time("2026-10-16T12:00:00Z"))[0]["fires"]


def test_book_rule_snapshot():
    from ncaaf_tracker.collect import book_rule_snapshot
    games = [{"espn_id": "1", "kickoff": "2026-10-17T19:30:00Z", "home_name": "Tennessee Volunteers",
              "away_name": "Alabama Crimson Tide", "home_short": "Tennessee", "away_short": "Alabama",
              "home_location": "Tennessee", "away_location": "Alabama", "home_abbr": "TENN", "away_abbr": "ALA",
              "home_id": "2633", "away_id": "333", "home_rank": 14, "away_rank": 6}]
    mk = [{"series": "KXNCAAFGAME", "event_ticker": "E", "ticker": "E-TENN", "yes_sub_title": "Tennessee",
           "kickoff_est": "2026-10-17T19:30:00Z", "yes_bid": 0.38, "yes_ask": 0.40},
          {"series": "KXNCAAFGAME", "event_ticker": "E", "ticker": "E-ALA", "yes_sub_title": "Alabama",
           "kickoff_est": "2026-10-17T19:30:00Z", "yes_bid": 0.60, "yes_ask": 0.62}]
    books = [{"ticker": "E-ALA", "side": "yes", "price": 0.60, "size": 500},   # backs Alabama
             {"ticker": "E-TENN", "side": "no", "price": 0.60, "size": 100},   # backs Alabama
             {"ticker": "E-TENN", "side": "yes", "price": 0.38, "size": 400}]  # backs Tennessee
    base = {"event_id": "an:1", "commence_time": "2026-10-17T19:30:00Z", "home_team": "Tennessee Volunteers",
            "away_team": "Alabama Crimson Tide", "book": "draftkings"}
    odds = [{**base, "market": "spreads", "outcome": "Tennessee Volunteers", "point": 4.0, "price": -108},
            {**base, "market": "spreads", "outcome": "Alabama Crimson Tide", "point": -4.0, "price": -112},
            {**base, "market": "h2h", "outcome": "Tennessee Volunteers", "point": None, "price": 160}]
    r = book_rule_snapshot(games, mk, books, odds, parse_time("2026-10-17T18:00:00Z"))[0]
    assert r["away_contracts"] == 600 and r["home_contracts"] == 400
    assert r["fires"] and r["pick_side"] == "home" and r["pick_line"] == 4.0 and r["pick_ml"] == 160
    assert r["pick_is_dog"] and r["kalshi_home_p"] == 0.39


def test_split_rules_snapshot():
    from ncaaf_tracker.collect import split_rules_snapshot
    games = [{"espn_id": "1", "kickoff": "2026-10-24T19:30:00Z", "home_name": "Tennessee Volunteers",
              "away_name": "Alabama Crimson Tide", "home_short": "Tennessee", "away_short": "Alabama",
              "home_location": "Tennessee", "away_location": "Alabama", "home_abbr": "TENN", "away_abbr": "ALA",
              "home_id": "2633", "away_id": "333", "home_rank": None, "away_rank": None}]
    base = {"event_id": "an:1", "commence_time": "2026-10-24T19:30:00Z", "home_team": "Tennessee Volunteers",
            "away_team": "Alabama Crimson Tide", "market": "spreads"}
    odds = [{**base, "book": "draftkings", "outcome": "Tennessee Volunteers", "point": 4.5, "price": -110},
            {**base, "book": "draftkings", "outcome": "Alabama Crimson Tide", "point": -4.5, "price": -110}]
    splits = [{**base, "side": "away", "tickets_pct": 75, "money_pct": 40},
              {**base, "side": "home", "tickets_pct": 25, "money_pct": 60}]
    r = {x["rule"]: x for x in split_rules_snapshot(games, odds, splits, parse_time("2026-10-24T12:00:00Z"))}
    assert r["sharp_split_v1"]["fires"] and not r["fade90_v1"]["fires"]
    assert r["sharp_split_v1"]["pick_side"] == "home" and r["sharp_split_v1"]["pick_line"] == 4.5
    splits[0].update(tickets_pct=92, money_pct=88); splits[1].update(tickets_pct=8, money_pct=12)
    r = {x["rule"]: x for x in split_rules_snapshot(games, odds, splits, parse_time("2026-10-24T12:00:00Z"))}
    assert r["fade90_v1"]["fires"] and not r["sharp_split_v1"]["fires"]
    splits[0]["money_pct"] = 100                       # bogus split -> no bet
    assert not any(x["fires"] for x in split_rules_snapshot(games, odds, splits, parse_time("2026-10-24T12:00:00Z")))


def test_vsin_parse():
    from ncaaf_tracker.sources import vsin
    page = """<html><body><div class="sp-empty-not">x</div>
    <table class="sp-table freezetable">
      <tr><th>10/31 12:00 PM</th><th>Spread</th><th>Handle %</th><th>Bet %</th><th>ML</th><th>Handle %</th>
          <th>Bet %</th><th>Total</th><th>Handle %</th><th>Bet %</th></tr>
      <tr><td><a class="sp-team-link">(12) Ole Miss</a></td><td>-13.5</td><td>29%</td><td>33%</td><td>-550</td>
          <td>60%</td><td>81%</td><td>o58.5</td><td>44%</td><td>61%</td></tr>
      <tr><td><a class="sp-team-link">Auburn</a></td><td>+13.5</td><td>71%</td><td>67%</td><td>+400</td>
          <td>40%</td><td>19%</td><td>u58.5</td><td>56%</td><td>39%</td></tr>
    </table></body></html>"""
    rows = vsin.parse(page, "CIRCA", "2026-10-31T12:00:00Z")
    assert len(rows) == 2
    a, h = rows
    assert a["side"] == "away" and a["team"] == "Ole Miss" and a["opponent"] == "Auburn"
    assert a["spread"] == -13.5 and a["spread_handle_pct"] == 29 and a["spread_bets_pct"] == 33
    assert a["ml"] == -550 and a["ml_handle_pct"] == 60 and a["total"] == 58.5 and a["total_bets_pct"] == 61
    assert h["spread"] == 13.5 and h["spread_handle_pct"] == 71 and h["source"] == "CIRCA"
    assert vsin.parse("<p>No splits data available for Tuesday</p>", "DK", "x") == []


def test_dkn_parse():
    from ncaaf_tracker.sources import dkn
    def odd(name, price, h, b):
        return (f'<div class="tb-sodd flex"><div class="tb-slipline flex-1 font-medium">{name}</div>'
                f'<div class="flex-1"><a class="tb-odd-s bg" href="x">{price}</a></div>'
                f'<div class="flex-1">{h}% <div class="tb-progress"><div style="width: {h}%"></div></div></div>'
                f'<div class="flex-1">{b}% <div class="tb-progress"><div style="width: {b}%"></div></div></div></div>')
    def head(m):
        return (f'<div class="tb-se-head flex"><div class="flex-1">{m}</div><div class="flex-1">Odds</div>'
                f'<div class="flex-1">% Handle</div><div class="flex-1">% Bets</div></div>')
    page = ('<div class="tb-se border-b"><div class="tb-se-title flex"><h5 class="mt-0">'
            '<a target="_blank" href="https://sportsbook.draftkings.com/event/34876703?x=1" class="no-underline">'
            ' UConn @ Air Force </a></h5><span class="t">10/31, 03:00PM </span></div><div class="tb-market-wrap">'
            + head("Moneyline") + odd("Air Force", "−125", 35, 30) + odd("UConn", "+105", 65, 70)
            + head("Spread") + odd("Air Force -2.5", "−105", 84, 57) + odd("UConn +2.5", "−115", 16, 43)
            + head("Total") + odd("Over 53.5", "−105", 64, 61) + odd("Under 53.5", "−115", 36, 39) + "</div></div>")
    rows = dkn.parse(page, "2026-10-31T18:00:00Z")
    assert len(rows) == 6
    sp = [r for r in rows if r["market"] == "spread"]
    assert sp[0]["side"] == "home" and sp[0]["line"] == -2.5 and sp[0]["odds"] == -105
    assert sp[0]["handle_pct"] == 84 and sp[0]["bets_pct"] == 57
    assert sp[1]["selection"] == "UConn" and sp[1]["side"] == "away" and sp[1]["line"] == 2.5
    assert rows[0]["dk_event_id"] == "34876703" and rows[0]["odds"] == -125 and rows[0]["line"] is None
    assert rows[4]["side"] == "over" and rows[4]["line"] == 53.5


def test_dkn_parse_nov2026_layout():
    from ncaaf_tracker.sources import dkn
    def odd(name, price, h, b):
        return (f'<div class="tb-sodd grid text-sm"><div class="tb-slipline min-w-0 font-medium">{name}</div>'
                f'<div><a target="_blank" class="tb-odd-s inline-block" href="x">\n{price}\n</a></div>'
                f'<div>{h}% <div class="tb-progress"><div style="width: {h}%"></div></div></div>'
                f'<div>{b}% <div class="tb-progress"><div style="width: {b}%"></div></div></div></div>')
    def mkt(label, a, b):
        return f'<div class="py-2.5"><div class="uppercase text-brand-orange">\n{label}\n</div><div class="tb-sm">{a}{b}</div></div>'
    page = ('<div class="tb-se rounded-lg p-3"><div class="tb-se-title flex"><h5 class="text-neutral-900">'
            '<a target="_blank" href="https://sportsbook.draftkings.com/event/34913140?x=1" class="no-underline">\n'
            'Ohio @ Akron </a></h5><span class="shrink-0">Tue 11/3 · 7:00 PM ET</span></div>'
            '<div class="tb-se-head grid uppercase"><div>Selection</div><div>Odds</div><div>Handle</div><div>Bets</div></div>'
            '<div class="tb-market-wrap">'
            + mkt("Moneyline", odd("Akron", "+410", 20, 8), odd("Ohio", "−550", 80, 92))
            + mkt("Spread", odd("Ohio -12.5", "−112", 95, 85), odd("Akron +12.5", "−108", 5, 15))
            + mkt("Total", odd("Over 51.5", "−112", 98, 89), odd("Under 51.5", "−108", 2, 11)) + "</div></div>")
    rows = dkn.parse(page, "2026-11-03T18:00:00Z")
    assert [r["market"] for r in rows] == ["moneyline"] * 2 + ["spread"] * 2 + ["total"] * 2
    assert rows[0]["selection"] == "Akron" and rows[0]["side"] == "home" and rows[0]["odds"] == 410
    assert rows[2]["side"] == "away" and rows[2]["line"] == -12.5 and rows[2]["handle_pct"] == 95 and rows[2]["bets_pct"] == 85
    assert rows[3]["line"] == 12.5 and rows[4]["line"] == 51.5 and rows[0]["game_time_et"] == "Tue 11/3 · 7:00 PM ET"
