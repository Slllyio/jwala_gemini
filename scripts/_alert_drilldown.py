"""Drill down on high-priority alerts in alerts_log."""
import psycopg2, yaml, sys

cfg = yaml.safe_load(open("config.yaml"))["database"]
conn = psycopg2.connect(
    host=cfg["host"], port=cfg["port"],
    dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"]
)
cur = conn.cursor()

# Show all columns
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name='alerts_log' ORDER BY ordinal_position
""")
cols = [r[0] for r in cur.fetchall()]
print("Columns:", cols)
print()

# Clearing alerts (most critical)
print("=" * 70)
print("CLEARING alerts (6 total)")
print("=" * 70)
cur.execute("""
    SELECT * FROM alerts_log
    WHERE change_type = 'Clearing'
    ORDER BY confidence DESC NULLS LAST
""")
rows = cur.fetchall()
for row in rows:
    for col, val in zip(cols, row):
        if col not in ("geom", "geometry"):
            print(f"  {col:25s}: {val}")
    print("-" * 50)

# Built expansion alerts
print()
print("=" * 70)
print("BUILT EXPANSION alerts (11 total)")
print("=" * 70)
cur.execute("""
    SELECT * FROM alerts_log
    WHERE change_type = 'Built expansion'
    ORDER BY confidence DESC NULLS LAST
    LIMIT 11
""")
rows = cur.fetchall()
for row in rows:
    for col, val in zip(cols, row):
        if col not in ("geom", "geometry"):
            print(f"  {col:25s}: {val}")
    print("-" * 50)

conn.close()
