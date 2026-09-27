# ncaaf

Tracks NCAAF sportsbook lines from the moment they appear until kickoff, alongside
Kalshi and Polymarket prices, order books and trade flow, plus final scores. The
goal is to test whether line moves line up with prediction-market money.

## What gets collected

No API keys needed. Sportsbook lines come from the public feeds that betting sites' own pages load.

| Source | What | Table(s) |
|---|---|---|
| ESPN scoreboard | schedule, kickoff, final score / winner, DraftKings open & close | `espn_games` |
| ESPN (finished games) | one row per completed game: home team, away team, both scores, winner, loser, winner side (home/away), margin, total. Postponed/canceled games excluded | `results` |
| Action Network (hourly) | DraftKings, FanDuel, BetMGM, Caesars, BetRivers, bet365 moneyline / spread / total, the **market opener** (`an_open`), and **public betting splits** (% of tickets and % of money per side) | `an_odds`, `an_splits` |
| Action Network (daily + backfill) | **exact line history**: every line change per book with a timestamp, opener to kickoff | `an_history` |
| Pinnacle web feed (hourly) | the sharp reference line | `pinnacle_odds` |
| Kalshi | `KXNCAAFGAME` / `KXNCAAFSPREAD` / `KXNCAAFTOTAL` quotes, volume, OI; moneyline order book (top 10) + trades with taker side | `kalshi_markets`, `kalshi_books`, `kalshi_trades` |
| Polymarket | CFB moneyline / spread / total markets: price, volume, liquidity; moneyline order book + trades with side | `poly_markets`, `poly_books`, `poly_trades` |
| The Odds API (optional) | only used if the `ODDS_API_KEY` secret is set | `odds` |

- **Earliest lines:** Action Network's history starts at each book's opener, so
  openers are exact even for games that were listed before tracking began.
  Hourly polling covers the current week and the next two.
- **Stops at kickoff:** live lines are skipped, history is cut at kickoff, and the
  analysis trims everything to the exact ESPN kickoff.
- **Opening/closing lines:** the per-book opener is the history's `opener` entry.
  The closer is the last line before kickoff. `market_open_*` is Action Network's
  market-opener line.
- **Money signals** to compare against line moves: Kalshi and Polymarket taker
  flow ($) and order-book imbalance, plus Action Network's public money% vs
  tickets%. Money% well above tickets% on a side means bigger bettors are on it.

These feeds are unofficial. They can change or block access without notice.
`collect.py --probe` (workflow mode `probe`) saves raw responses from every
source to `probe/` on the data branch so a broken parser can be fixed against
the new format.

Data is stored append-only as `data/<table>/<date>/<table>_<timestamp>.csv.gz` on
the **`data` branch**, one file per table per run, which keeps the code branch clean.

## Setup (one time)

1. Merge this to `main`. GitHub only runs scheduled workflows from the default branch.
2. (Optional) add an Odds API key as the repo secret `ODDS_API_KEY` for extra books.
   Everything works without it.
3. Settings → Actions → General → Workflow permissions → **Read and write**.
4. Actions → `collect` → Run workflow with `mode=backfill`, `since=2026-08-22`.
   This loads earlier games' results, ESPN open/close lines, and Action Network's
   full line history for every game played so far.
5. Actions → `collect` → Run workflow with `mode=full` to confirm everything works.
   The `runs` table logs per-source row counts and errors, and the job log shows
   warnings if a source returns nothing.

**Actions minutes:** about 720 hourly runs plus about 500 short closing runs a month
comes to roughly 2,000–2,500 minutes, which is at or over the 2,000 free minutes
for a private repo. Either make the repo public (Actions is free) or trim the
weeknight crons in `.github/workflows/collect.yml`. A $5 VPS running
`python -m ncaaf_tracker.collect` from cron (`7 * * * *` plus the closing crons)
also works.

## Analysis

```bash
pip install -r requirements-analysis.txt
git fetch origin data && git worktree add store origin/data
python -m ncaaf_tracker.analyze --data-dir store/data --out out
```

The `analyze` workflow also runs this every day. It uploads the full output as an
artifact and commits `report.md` and `games_summary.csv` to `data:summary/`.

| File | Contents |
|---|---|
| `games_summary.csv` | one row per game: winner, margin, cover / O-U vs the closing line, ESPN open/close, tracked consensus + Pinnacle open/close, Kalshi/Polymarket open/close probability, volume, net $ flow |
| `book_open_close.csv` | game × book × market: first line, closing line, move, number of changes |
| `line_history.csv` | every book snapshot (home spread, total, ML, devigged P(home)) |
| `hourly_panel.csv` | game × hour before kickoff: consensus line, Pinnacle, Kalshi/Poly P(home), volume, net taker flow ($), resting-book imbalance |
| `unmatched.csv` | source games that couldn't be tied to an ESPN game. Add a row to `aliases.csv` (`raw_name,espn_team_id`) to fix |
| `report.md` | same-hour correlations, lead/lag (which side moves first), whether big book moves had PM money behind them, Brier score by source, and whether following the move beat the opener |

Conventions: spreads are the **home** team's line; probabilities are P(home wins)
with the vig removed; flow > 0 means money went on the home team.

## Notes / limits

- The raw order book shows resting interest, not money that moved. The
  "money" signal is **trade flow by taker side** (`*_flow`), with volume deltas as
  a fallback. Book imbalance is kept as a secondary signal.
- Kalshi and Polymarket college football volume is thin outside marquee games.
  Expect most correlation power to come from the top ~50 games per week.
- Kalshi spread/total series tickers and Polymarket's CFB series are discovered or
  configured, not guaranteed. Run
  `python -m ncaaf_tracker.collect --discover` to list them. Override with
  `KALSHI_SERIES` or `POLY_SERIES_IDS` if either platform renames them.
- Other exchanges: Novig and ProphetX come in through The Odds API's `us_ex`
  region. Betfair has essentially no NCAAF liquidity and isn't usable from the US.
  Sporttrade, Crypto.com and the other US event-contract apps have no public
  market-data API.

## Tests

```bash
pip install -r requirements-analysis.txt && pytest -q
```
