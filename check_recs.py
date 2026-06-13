import json
import sqlite3

db = sqlite3.connect('var/omega_traces.db')
c = db.cursor()

c.execute("SELECT t.trace_id FROM bet_ledger b JOIN traces t ON t.trace_id = b.trace_id")
ledger_trace_ids = set(row[0] for row in c.fetchall())

c.execute("SELECT trace_id, league, full_trace FROM traces WHERE league IN ('MLB', 'NBA')")
traces = c.fetchall()

missing_traces = []
missing_with_rec = []
bad_rows = []
for row in traces:
    if row[0] not in ledger_trace_ids:
        missing_traces.append(row)
        try:
            t_data = json.loads(row[2])
        except (json.JSONDecodeError, TypeError) as exc:
            bad_rows.append((row[0], str(exc)))
            continue
        res = t_data.get('result', {})
        if res and res.get('recommendation') is not None and res.get('recommendation') != 'PASS':
            missing_with_rec.append(row)

print(f"Total traces missing from ledger: {len(missing_traces)}")
print(f"Traces missing from ledger WITH recommendation: {len(missing_with_rec)}")
print(f"Malformed trace rows skipped: {len(bad_rows)}")
