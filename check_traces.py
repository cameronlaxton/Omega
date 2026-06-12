import sqlite3
import json

db = sqlite3.connect('var/omega_traces.db')
c = db.cursor()

# Find traces for MLB/NBA
c.execute("SELECT trace_id, league, full_trace FROM traces WHERE league IN ('MLB', 'NBA')")
traces = c.fetchall()

print(f"Total MLB/NBA traces: {len(traces)}")

# Find ledger entries for MLB/NBA
c.execute("SELECT t.trace_id FROM bet_ledger b JOIN traces t ON t.trace_id = b.trace_id WHERE t.league IN ('MLB', 'NBA')")
ledger_trace_ids = set(row[0] for row in c.fetchall())

print(f"Traces with bet ledger: {len(ledger_trace_ids)}")

missing = [t for t in traces if t[0] not in ledger_trace_ids]
print(f"Traces without bet ledger: {len(missing)}")

if missing:
    for row in missing[:2]:
        print(row[0], row[1])
        t_data = json.loads(row[2])
        print(json.dumps(t_data.get('result', {}), indent=2))
        print(json.dumps(t_data.get('input_snapshot', {}), indent=2))

