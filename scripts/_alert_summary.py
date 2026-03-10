"""Print a formatted alert summary table from PostGIS alerts_log."""
import pathlib, sys, yaml, psycopg2

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("database", {})
conn = psycopg2.connect(
    host=cfg.get("host", "localhost"),
    port=cfg.get("port", 5432),
    dbname=cfg.get("dbname", "gis_projects"),
    user=cfg.get("user", "postgres"),
    password=cfg.get("password", ""),
)
cur = conn.cursor()

RUN_DATE = "2026-02-24"

# ── SECTION 1: Overall counts ─────────────────────────────────────────────────
cur.execute("""
SELECT
    COUNT(*) total,
    COUNT(DISTINCT beat_name) beats,
    COUNT(CASE WHEN stacked_score>=0.70 THEN 1 END) high,
    COUNT(CASE WHEN stacked_score>=0.45 AND stacked_score<0.70 THEN 1 END) med,
    COUNT(CASE WHEN stacked_score>=0.25 AND stacked_score<0.45 THEN 1 END) low,
    ROUND(SUM(area_ha)::numeric,1) total_ha,
    MIN(detection_date)::date earliest,
    MAX(detection_date)::date latest
FROM alerts_log
WHERE ingested_at::date = %s
""", (RUN_DATE,))
ov = cur.fetchone()
print()
print("=" * 75)
print(f"  EWS ALERT SUMMARY — Live Run {RUN_DATE}")
print("=" * 75)
print(f"  Total alerts : {ov[0]:,}")
print(f"  Beats with alerts: {ov[1]}")
print(f"  HIGH (>=0.70): {ov[2]:,}   MEDIUM (0.45-0.70): {ov[3]:,}   LOW (0.25-0.45): {ov[4]:,}")
print(f"  Total affected area: {ov[5]} ha")
print(f"  Detection window : {ov[6]}  ->  {ov[7]}  (t0 -> t1)")
print("=" * 75)

# ── SECTION 2: By threat type ─────────────────────────────────────────────────
cur.execute("""
SELECT change_type,
    COUNT(*) n,
    ROUND(AVG(stacked_score)::numeric,3) avg_s,
    ROUND(MAX(stacked_score)::numeric,3) top_s,
    COUNT(CASE WHEN stacked_score>=0.70 THEN 1 END) hi,
    COUNT(CASE WHEN stacked_score>=0.45 AND stacked_score<0.70 THEN 1 END) md,
    COUNT(CASE WHEN stacked_score>=0.25 AND stacked_score<0.45 THEN 1 END) lo,
    ROUND(SUM(area_ha)::numeric,1) ha
FROM alerts_log
WHERE ingested_at::date = %s
GROUP BY change_type ORDER BY n DESC
""", (RUN_DATE,))
rows = cur.fetchall()
print()
print("BY THREAT TYPE")
print(f"{'Threat':<22} {'# Alerts':>9} {'Avg Score':>10} {'Top Score':>10} {'HIGH':>6} {'MED':>6} {'LOW':>6} {'Area (ha)':>10}")
print("-" * 81)
for r in rows:
    print(f"{str(r[0] or 'unknown'):<22} {r[1]:>9} {str(r[2]):>10} {str(r[3]):>10} {r[4]:>6} {r[5]:>6} {r[6]:>6} {str(r[7]):>10}")

# ── SECTION 3: Beat-level with date pairs ─────────────────────────────────────
cur.execute("""
SELECT
    beat_name, change_type,
    COUNT(*) n,
    ROUND(MAX(stacked_score)::numeric,3) top_s,
    COUNT(CASE WHEN stacked_score>=0.70 THEN 1 END) hi,
    COUNT(CASE WHEN stacked_score>=0.45 AND stacked_score<0.70 THEN 1 END) md,
    COUNT(CASE WHEN stacked_score>=0.25 AND stacked_score<0.45 THEN 1 END) lo,
    ROUND(SUM(area_ha)::numeric,1) ha,
    detection_date::date t1,
    detection_period period
FROM alerts_log
WHERE ingested_at::date = %s
GROUP BY beat_name, change_type, detection_date::date, detection_period
ORDER BY n DESC
""", (RUN_DATE,))
rows2 = cur.fetchall()

print()
print(f"BEAT-LEVEL ALERTS WITH DATE PAIRS  ({len(rows2)} rows)")
print(f"{'Beat':<26} {'Threat':<20} {'N':>4} {'TopS':>5} {'H':>4} {'M':>4} {'L':>4} {'Ha':>8}  {'t0 (reference)':>14}  {'t1 (observed)':>13}")
print("-" * 110)
for r in rows2:
    beat, threat, n, top, hi, md, lo, ha, t1, period = r
    if period and " to " in str(period):
        parts = str(period).split(" to ")
        t0_s = parts[0].strip()[:10]
        t1_s = parts[1].strip()[:10]
    else:
        t0_s = "unknown"
        t1_s = str(t1) if t1 else "unknown"
    print(f"{str(beat)[:26]:<26} {str(threat)[:20]:<20} {n:>4} {str(top):>5} {hi:>4} {md:>4} {lo:>4} {str(ha):>8}  {t0_s:>14}  {t1_s:>13}")

# ── SECTION 4: Top 25 HIGH severity individual spots ─────────────────────────
cur.execute("""
SELECT
    beat_name, change_type,
    ROUND(stacked_score::numeric,3) score,
    CASE WHEN stacked_score>=0.70 THEN 'HIGH'
         WHEN stacked_score>=0.45 THEN 'MEDIUM' ELSE 'LOW' END label,
    ROUND(area_ha::numeric,2) ha,
    ROUND(mean_delta_trees::numeric,3) d_trees,
    ROUND(mean_delta_crops::numeric,3) d_crops,
    centroid_lat, centroid_lon,
    detection_date::date t1,
    detection_period period
FROM alerts_log
WHERE ingested_at::date = %s AND stacked_score >= 0.70
ORDER BY stacked_score DESC, area_ha DESC LIMIT 25
""", (RUN_DATE,))
rows3 = cur.fetchall()

print()
print("TOP 25 HIGH-SEVERITY ALERT SPOTS (score >= 0.70)")
print(f"{'Beat':<26} {'Threat':<18} {'Score':>6} {'Label':>7} {'Ha':>8} {'dTrees':>8} {'dCrops':>8}  Lat        Lon        t0->t1")
print("-" * 120)
for r in rows3:
    beat, threat, score, label, ha, dt, dc, lat, lon, t1, period = r
    if period and " to " in str(period):
        parts = str(period).split(" to ")
        pair = f"{parts[0].strip()[:10]} -> {parts[1].strip()[:10]}"
    else:
        pair = f"? -> {str(t1)}"
    lat_s = f"{lat:.4f}" if lat else "?"
    lon_s = f"{lon:.4f}" if lon else "?"
    print(f"{str(beat)[:26]:<26} {str(threat)[:18]:<18} {str(score):>6} {str(label):>7} {str(ha):>8} {str(dt):>8} {str(dc):>8}  {lat_s:<10} {lon_s:<10} {pair}")

conn.close()
print()
print("Done.")
