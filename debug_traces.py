import sqlite3
import json

db = sqlite3.connect('var/omega_traces.db')
c = db.cursor()

c.execute("SELECT t.trace_id FROM bet_ledger b JOIN traces t ON t.trace_id = b.trace_id")
ledger_trace_ids = set(row[0] for row in c.fetchall())

c.execute("SELECT trace_id, league, timestamp, full_trace FROM traces WHERE league IN ('MLB', 'NBA')")
traces = c.fetchall()

missing = [t for t in traces if t[0] not in ledger_trace_ids]
valid = 0
for trace_id, league, timestamp, full_trace in missing:
    t_data = json.loads(full_trace)
    rec = t_data.get('recommendation')
    if not rec:
        rec = t_data.get('result', {}).get('recommendation')
    if rec and rec != 'PASS':
        valid += 1
        print(f"Found one! {trace_id} rec={rec}")
        
print(f"Total valid: {valid}")
