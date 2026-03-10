"""Print alerts_log schema and a sample row."""
import psycopg2, yaml
from pathlib import Path

cfg = yaml.safe_load(Path('config.yaml').read_text())['database']
conn = psycopg2.connect(
    host=cfg['host'], port=cfg['port'],
    dbname=cfg['dbname'], user=cfg['user'],
    password=cfg.get('password', '')
)
cur = conn.cursor()

# schema
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name='alerts_log'
    ORDER BY ordinal_position
""")
print("=== alerts_log columns ===")
for row in cur.fetchall():
    print(f"  {row[0]:<30} {row[1]}")

# sample
print("\n=== Sample row (dict) ===")
cur.execute("SELECT * FROM alerts_log WHERE geom IS NOT NULL LIMIT 1")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
if row:
    for k, v in zip(cols, row):
        if k not in ('geom',):
            print(f"  {k:<30} {v}")

print("\n=== Count by change_type ===")
cur.execute("""
    SELECT change_type, COUNT(*), round(AVG(COALESCE(confidence, stacked_score))::numeric, 3)
    FROM alerts_log
    GROUP BY change_type
    ORDER BY COUNT(*) DESC
""")
for r in cur.fetchall():
    print(f"  {str(r[0]):<25} n={r[1]:5d}  avg_conf={r[2]}")

conn.close()
