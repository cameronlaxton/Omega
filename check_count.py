import sqlite3
import json

db = sqlite3.connect('var/omega_traces.db')
c = db.cursor()

c.execute("""
    SELECT count(*)
    FROM bet_ledger b
    JOIN traces t ON t.trace_id = b.trace_id
    LEFT JOIN closing_lines c ON c.trace_id = b.trace_id AND c.market = b.market AND c.selection_descriptor = b.selection_descriptor
    WHERE t.league IN ('MLB', 'NBA') AND c.closing_id IS NULL
""")
print(f"Pending MLB/NBA bets needing close: {c.fetchone()[0]}")

c.execute("""
    SELECT count(*)
    FROM bet_ledger b
    JOIN traces t ON t.trace_id = b.trace_id
    WHERE t.league IN ('MLB', 'NBA') AND b.status = 'pending'
""")
print(f"Total pending MLB/NBA bets: {c.fetchone()[0]}")
