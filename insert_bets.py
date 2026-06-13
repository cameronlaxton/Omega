import json
import uuid

from omega.trace.bet_record import BetRecord
from omega.trace.store import TraceStore

db_path = 'var/omega_traces.db'
store = TraceStore(db_path=db_path)

c = store.conn.cursor()

c.execute("SELECT t.trace_id FROM bet_ledger b JOIN traces t ON t.trace_id = b.trace_id")
ledger_trace_ids = set(row[0] for row in c.fetchall())

c.execute("SELECT trace_id, league, timestamp, full_trace FROM traces WHERE league IN ('MLB', 'NBA')")
traces = c.fetchall()

inserted = 0
skipped_missing_timestamp = []
skipped_missing_odds = []

for trace_id, league, timestamp, full_trace in traces:
    if trace_id in ledger_trace_ids:
        continue

    t_data = json.loads(full_trace)

    # Check both top level and result dict
    rec = t_data.get('recommendation')
    if not rec:
        res = t_data.get('result', {})
        rec = res.get('recommendation')

    if not rec or rec == 'PASS':
        continue

    trace_timestamp = timestamp or t_data.get('timestamp')
    if not trace_timestamp:
        skipped_missing_timestamp.append(trace_id)
        continue

    snap = t_data.get('input_snapshot', {})
    if not snap:
        snap = t_data # flat trace

    kind = t_data.get('kind', 'game')
    if 'prop_type' in t_data:
        kind = 'prop'

    # Extract line, odds based on recommendation
    if kind == 'prop':
        market = f"player_prop:{t_data.get('prop_type', 'unknown')}"
        line = t_data.get('line')

        if rec == 'OVER':
            sel_desc = 'over'
            odds = t_data.get('odds_over')
            selection = f"{t_data.get('player_name')} Over {line}"
        elif rec == 'UNDER':
            sel_desc = 'under'
            odds = t_data.get('odds_under')
            selection = f"{t_data.get('player_name')} Under {line}"
        else:
            continue
    else:
        # Game trace
        if 'spread' in rec.lower():
            market = 'spread'
            if 'home' in rec.lower():
                sel_desc = 'home_spread_line'
                odds = snap.get('home_spread_odds')
                line = snap.get('home_spread')
                selection = f"{snap.get('home_team')} {line}"
            else:
                sel_desc = 'away_spread_line'
                odds = snap.get('away_spread_odds')
                line = snap.get('away_spread')
                selection = f"{snap.get('away_team')} {line}"
        elif 'total' in rec.lower():
            market = 'total'
            line = snap.get('total')
            if 'over' in rec.lower():
                sel_desc = 'total_over_line'
                odds = snap.get('total_over_odds')
                selection = f"Over {line}"
            else:
                sel_desc = 'total_under_line'
                odds = snap.get('total_under_odds')
                selection = f"Under {line}"
        else:
            market = 'moneyline'
            line = None
            if 'home' in rec.lower():
                sel_desc = 'home_moneyline'
                odds = snap.get('home_moneyline')
                selection = snap.get('home_team')
            else:
                sel_desc = 'away_moneyline'
                odds = snap.get('away_moneyline')
                selection = snap.get('away_team')

    if odds is None:
        # Fallback to result if missing in snap
        if 'bet_side_odds' in t_data and t_data['bet_side_odds']:
            odds = t_data['bet_side_odds']
        else:
            skipped_missing_odds.append(trace_id)
            continue

    block = {
        "book": "consensus",
        "market": market,
        "selection": selection,
        "selection_descriptor": sel_desc,
        "line_taken": line,
        "odds_taken": odds,
        "units": t_data.get('recommended_units', 1.0),
        "decision_timestamp": trace_timestamp
    }

    bet_id = uuid.uuid4().hex[:12]
    bet = BetRecord.from_export_block(trace_id=trace_id, bet_id=bet_id, block=block)
    bet.provenance = "engine_auto"

    store.record_bet(bet)
    inserted += 1

store.close()
print(f"Inserted {inserted} bets.")
print(f"Skipped missing timestamp: {len(skipped_missing_timestamp)}")
print(f"Skipped missing odds: {len(skipped_missing_odds)}")
