# ncaaf

Tracks NCAAF sportsbook lines from the moment they appear until kickoff, alongside
Kalshi and Polymarket prices, order books and trade flow, plus final scores. The
goal is to test whether line moves line up with prediction-market money.

## What gets collected

| Source | What | Key | Table(s) |
|---|---|---|---|
| ESPN scoreboard | schedule, kickoff, final score / winner, ESPN's own **open & close** spread / total / ML | none | `espn_games` |
| [The Odds API](https://the-odds-api.com) | every US book (DK, FD, MGM, Caesars, ...), Pinnacle (`eu`), Novig / ProphetX exchanges (`us_ex`): moneyline, spread, total | `ODDS_API_KEY` | `odds` |
| Kalshi | `KXNCAAFGAME` / `KXNCAAFSPREAD` / `KXNCAAFTOTAL` quotes, volume, OI; moneyline order book (top 10) + trades with taker side | none | `kalshi_markets`, `kalshi_books`, `kalshi_trades` |
| Polymarket | CFB game markets: price, volume, liquidity; moneyline order book (top 10) + trades with side | none | `poly_markets`, `poly_books`, `poly_trades` |

- **Earliest lines:** the hourly run pulls every game any book has posted, with no
  upper date limit, so a line gets captured within an hour of being posted.
- **Stops at kickoff:** sportsbook rows are filtered to games that haven't started.
  Prediction-market kickoff times are estimates, so those are collected with 1h of
  slack, and the analysis step trims everything to the exact ESPN kickoff.
- **Opening/closing lines:** opener = first snapshot seen, closer = last snapshot
  before kickoff, for each book and for the consensus. ESPN's provider open/close
  is stored separately. Games whose line already existed when tracking started are
  flagged `opener_censored`.
- **Closing-line precision:** on top of the hourly snapshot there are extra
  15-minute runs in the Saturday and weeknight kickoff windows. Those runs poll
  only games kicking off in the next 100 minutes, so the closing line is at most
  about 15 minutes old.

Data is stored append-only as `data/<table>/<date>/<table>_<timestamp>.csv.gz` on
the **`data` branch**, one file per table per run, which keeps the code branch clean.

## Setup (one time)

1. Merge this to `main`. GitHub only runs scheduled workflows from the default branch.
2. Get an Odds API key and add it as the repo secret `ODDS_API_KEY`
   (Settings → Secrets and variables → Actions).
   - Hourly polling across 3 markets × 3 regions uses about **9 credits per call,
     roughly 11k credits a month**. That fits the 20K plan; the 500-credit free
     tier can't support hourly polling.
   - To cut cost, set the repo variable `ODDS_API_REGIONS=us` (3 credits a call).
     This drops Pinnacle, the sharpest reference line.
3. Settings → Actions → General → Workflow permissions → **Read and write**.
4. Actions → `collect` → Run workflow with `mode=backfill`, `since=2026-08-22`.
   This loads earlier games' results and ESPN open/close lines.
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
