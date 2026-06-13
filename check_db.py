import json
import logging
import sqlite3

db=sqlite3.connect('var/omega_traces.db')
c=db.cursor()
c.execute("SELECT b.ledger_id, b.market, b.selection, b.selection_descriptor, b.line, b.odds, t.full_trace FROM bet_ledger b JOIN traces t ON t.trace_id = b.trace_id WHERE b.status = 'pending'")

logger = logging.getLogger(__name__)
for row in c.fetchall():
    try:
        t = json.loads(row[6])
        home = t.get('input_snapshot', {}).get('home_team', '')
        away = t.get('input_snapshot', {}).get('away_team', '')
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.debug("failed to parse trace teams for ledger_id=%s: %s", row[0], exc)
        home, away = '', ''
    print(f"ledger_id={row[0]} market={row[1]} sel={row[2]} desc={row[3]} line={row[4]} odds={row[5]} home={home} away={away}")
