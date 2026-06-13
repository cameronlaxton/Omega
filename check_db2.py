import sqlite3
db=sqlite3.connect('var/omega_traces.db')
c=db.cursor()
c.execute("SELECT b.ledger_id, b.market, b.provenance, t.league FROM bet_ledger b JOIN traces t ON t.trace_id = b.trace_id WHERE b.status = 'pending'")
for row in c.fetchall():
    print(row)
