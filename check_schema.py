import sqlite3
db = sqlite3.connect('var/omega_traces.db')
c = db.cursor()
c.execute("PRAGMA table_info(bet_ledger)")
for row in c.fetchall():
    print(row)
