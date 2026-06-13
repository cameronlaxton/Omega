import sqlite3

db = sqlite3.connect('var/omega_traces.db')
c = db.cursor()

c.execute("""
    SELECT count(*)
    FROM bet_ledger b
    JOIN traces t ON t.trace_id = b.trace_id
    LEFT JOIN closing_lines cl
      ON cl.trace_id = b.trace_id
     AND cl.market = b.market
     AND (
        cl.selection_descriptor = b.selection_descriptor
        OR (cl.selection_descriptor IS NULL AND b.selection_descriptor IS NULL)
     )
    WHERE t.league IN ('MLB', 'NBA')
      AND cl.closing_id IS NULL
""")
print(f"Missing closing lines for MLB/NBA: {c.fetchone()[0]}")

c.execute("""
    SELECT b.status, count(*)
    FROM bet_ledger b
    JOIN traces t ON t.trace_id = b.trace_id
    LEFT JOIN closing_lines cl
      ON cl.trace_id = b.trace_id
     AND cl.market = b.market
     AND (
        cl.selection_descriptor = b.selection_descriptor
        OR (cl.selection_descriptor IS NULL AND b.selection_descriptor IS NULL)
     )
    WHERE t.league IN ('MLB', 'NBA')
      AND cl.closing_id IS NULL
    GROUP BY b.status
""")
print("Breakdown by status:")
for row in c.fetchall():
    print(row)
