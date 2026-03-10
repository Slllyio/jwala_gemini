"""Quick script to verify psycopg2 imports and test the PostGIS connection."""
import os
import sys
try:
    import psycopg2
    print(f"psycopg2 version: {psycopg2.__version__}")
except ImportError as e:
    print(f"psycopg2 MISSING: {e}")
    sys.exit(1)

try:
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f).get("database", {})
    dsn = (
        f"host={cfg.get('host','localhost')} "
        f"port={cfg.get('port',5432)} "
        f"dbname={cfg.get('dbname','gis_projects')} "
        f"user={cfg.get('user','postgres')} "
        f"password={os.environ.get('VS_DB_PASSWORD', cfg.get('password', ''))}"
    )
    print(f"DSN (password masked): {dsn.replace(os.environ.get('VS_DB_PASSWORD', cfg.get('password', '')), '****')}")
    conn = psycopg2.connect(dsn, connect_timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print(f"DB connected OK: {cur.fetchone()[0][:60]}")
    # Check if alerts table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'alerts_log'
        )
    """)
    exists = cur.fetchone()[0]
    print(f"alerts_log table exists: {exists}")
    conn.close()
except Exception as e:
    print(f"DB connection FAILED: {e}")
    print("(Expected if PostgreSQL is not running; that is fine for dry-run mode.)")
    print("To enable live runs: ensure PostgreSQL is running with the config.yaml credentials.")
