"""
scripts/beat_report.py
======================
Generates a per-beat PDF summary report for field officers.

For every forest beat that has ≥1 alert in the current reporting window,
this script produces a single-page PDF containing:
  - Beat header: name, sub-range, alert count, total area lost
  - Risk bar:    colour-coded confidence band (Low / Medium / High / Critical)
  - SAR CuSum:   gradual-loss indicator and boost status
  - Alert table: top-10 alerts by confidence (date, area, confidence, SAR boost)
  - Simple map:  lat/lon scatter of alert centroids for the beat

All beats are also bundled into a single consolidated PDF
(outputs/reports/beat_report_<date>.pdf).

Usage
-----
    python scripts/beat_report.py --config config.yaml [--date 2024-06-01]
    python scripts/beat_report.py --config config.yaml --beat "Chanderi Beat"

Dependencies: psycopg, matplotlib, reportlab
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Colour thresholds (matching compute_zone_risk tiers) ─────────────────────
_TIERS = [
    (0.00, 0.10, "Low",      "#2ECC71"),
    (0.10, 0.20, "Moderate", "#F39C12"),
    (0.20, 0.40, "High",     "#E74C3C"),
    (0.40, 1.00, "Critical", "#922B21"),
]


def _tier(confidence: float) -> tuple[str, str]:
    """Return (label, hex_colour) for a confidence value."""
    for lo, hi, label, colour in _TIERS:
        if lo <= confidence < hi:
            return label, colour
    return "Critical", "#922B21"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _connect(cfg: dict):
    import psycopg
    db = cfg.get("database", {})
    return psycopg.connect(
        host=db.get("host", "localhost"),
        port=db.get("port", 5432),
        dbname=db.get("dbname", "gis_projects"),
        user=db.get("user", "postgres"),
        password=os.environ.get("VS_DB_PASSWORD", db.get("password", "")),
    )


def _fetch_beats(conn, date_from: str, date_to: str, beat_filter: str | None):
    """Return list of beat-name strings that have alerts in the window."""
    q = """
        SELECT DISTINCT beat_name
        FROM alerts_log
        WHERE beat_name IS NOT NULL
          AND detection_date BETWEEN %s AND %s
    """
    params = [date_from, date_to]
    if beat_filter:
        q += " AND beat_name = %s"
        params.append(beat_filter)
    q += " ORDER BY beat_name"
    cur = conn.cursor()
    cur.execute(q, params)
    return [r[0] for r in cur.fetchall()]


def _fetch_beat_alerts(conn, beat_name: str, date_from: str, date_to: str):
    """Return all alert rows for a beat in the window, sorted by confidence."""
    cur = conn.cursor()
    cur.execute(
        """SELECT id, detection_date, area_ha,
                  COALESCE(stacked_score, confidence) AS conf,
                  change_type, cusum_zone_score, sar_boost_applied,
                  centroid_lat, centroid_lon,
                  mean_delta_trees, mean_delta_crops, mean_delta_bare
           FROM alerts_log
           WHERE beat_name = %s
             AND detection_date BETWEEN %s AND %s
           ORDER BY conf DESC NULLS LAST""",
        [beat_name, date_from, date_to],
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ── matplotlib single-beat page ───────────────────────────────────────────────

def _make_beat_figure(beat_name: str, alerts: list[dict], date_from: str, date_to: str):
    """Return a matplotlib Figure for one beat (A4-ish aspect)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    fig.patch.set_facecolor("#1A1A2E")

    gs = GridSpec(4, 2, figure=fig,
                  left=0.08, right=0.95, top=0.93, bottom=0.06,
                  hspace=0.45, wspace=0.35)

    # ── Header ──────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    total_area = sum(a["area_ha"] or 0 for a in alerts)
    avg_conf   = (sum((a["conf"] or 0) for a in alerts) / max(len(alerts), 1))
    tier_lbl, tier_col = _tier(avg_conf)
    n_sar = sum(1 for a in alerts if a["sar_boost_applied"])

    ax_hdr.set_facecolor("#16213E")
    ax_hdr.text(0.01, 0.82, f"🌲 {beat_name}", fontsize=16, fontweight="bold",
                color="white", transform=ax_hdr.transAxes, va="top")
    ax_hdr.text(0.01, 0.45,
                f"Period: {date_from}  →  {date_to}  |  "
                f"Alerts: {len(alerts)}  |  "
                f"Area lost: {total_area:.1f} ha  |  "
                f"Avg risk: {avg_conf:.2f} ({tier_lbl})  |  "
                f"SAR-boosted: {n_sar}",
                fontsize=8, color="#B0B0C0", transform=ax_hdr.transAxes, va="top")

    # Risk colour bar
    bar_ax = ax_hdr.inset_axes([0.0, 0.0, 1.0, 0.12])
    bar_ax.axis("off")
    for lo, hi, lbl, col in _TIERS:
        bar_ax.barh(0, hi - lo, left=lo, color=col, height=1)
        bar_ax.text((lo + hi) / 2, 0, lbl, ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold")
    # marker for avg_conf
    bar_ax.axvline(avg_conf, color="white", linewidth=2, linestyle="--")

    # ── Map scatter ─────────────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[1, 0])
    ax_map.set_facecolor("#0F3460")
    lats  = [a["centroid_lat"] for a in alerts if a["centroid_lat"]]
    lons  = [a["centroid_lon"] for a in alerts if a["centroid_lon"]]
    confs = [a["conf"] or 0 for a in alerts if a["centroid_lat"]]
    sc = ax_map.scatter(lons, lats, c=confs, cmap="RdYlGn_r",
                        vmin=0, vmax=0.5, s=40, edgecolors="white", linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=ax_map, label="Confidence", shrink=0.9)
    ax_map.set_title("Alert Centroids", color="white", fontsize=9, pad=4)
    ax_map.tick_params(colors="gray", labelsize=6)
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#334")
    ax_map.set_facecolor("#0F3460")

    # ── CuSum bar ──────────────────────────────────────────────────────────
    ax_cs = fig.add_subplot(gs[1, 1])
    ax_cs.set_facecolor("#0F3460")
    cusum_vals = [a["cusum_zone_score"] for a in alerts if a["cusum_zone_score"] is not None]
    if cusum_vals:
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        ax_cs.hist(cusum_vals, bins=bins, color="#E67E22", edgecolor="white", linewidth=0.5)
        ax_cs.axvline(0.5, color="#F1C40F", linestyle="--", linewidth=1, label="Boost threshold")
        ax_cs.legend(fontsize=6, labelcolor="white", facecolor="#111")
    else:
        ax_cs.text(0.5, 0.5, "No CuSum data", ha="center", va="center",
                   color="gray", transform=ax_cs.transAxes)
    ax_cs.set_title("CuSum Score Distribution", color="white", fontsize=9, pad=4)
    ax_cs.tick_params(colors="gray", labelsize=6)
    ax_cs.set_facecolor("#0F3460")
    for spine in ax_cs.spines.values():
        spine.set_edgecolor("#334")

    # ── Delta radar / bar ───────────────────────────────────────────────────
    ax_delta = fig.add_subplot(gs[2, :])
    ax_delta.set_facecolor("#0F3460")
    delta_fields = ["mean_delta_trees", "mean_delta_crops", "mean_delta_bare"]
    delta_labels = ["Trees", "Crops", "Bare"]
    delta_means  = []
    for f in delta_fields:
        vals = [a[f] for a in alerts if a.get(f) is not None]
        delta_means.append(sum(vals) / max(len(vals), 1) if vals else 0.0)

    colours = ["#27AE60", "#F1C40F", "#95A5A6"]
    bars = ax_delta.barh(delta_labels, delta_means, color=colours, edgecolor="white", linewidth=0.4)
    ax_delta.axvline(0, color="white", linewidth=0.5)
    for bar, val in zip(bars, delta_means):
        ax_delta.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                      f"{val:+.3f}", va="center", fontsize=7, color="white")
    ax_delta.set_title("Mean LULC Deltas (avg over alerts)", color="white", fontsize=9, pad=4)
    ax_delta.tick_params(colors="white", labelsize=7)
    for spine in ax_delta.spines.values():
        spine.set_edgecolor("#334")

    # ── Alert table ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")
    top10   = alerts[:10]
    headers = ["ID", "Date", "Area (ha)", "Confidence", "Change", "SAR↑"]
    rows    = [
        [
            str(a["id"]),
            str(a["detection_date"])[:10] if a["detection_date"] else "—",
            f"{a['area_ha']:.2f}" if a["area_ha"] else "—",
            f"{a['conf']:.3f}"  if a["conf"]    else "—",
            (a["change_type"] or "—")[:12],
            "✓" if a["sar_boost_applied"] else "·",
        ]
        for a in top10
    ]

    tbl = ax_tbl.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#16213E" if r == 0 else "#0D1B2A")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#334")
    ax_tbl.set_title(f"Top {len(top10)} Alerts by Confidence", color="white",
                     fontsize=9, pad=4)

    fig.suptitle(f"Van Suraksha · Beat Report  |  {date.today():%d %b %Y}",
                 color="#AAB4BE", fontsize=9, y=0.97)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Van Suraksha Beat PDF Report Generator")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--date",     default=None,
                        help="Anchor date YYYY-MM-DD (default: today)")
    parser.add_argument("--days",     type=int, default=30,
                        help="Look-back window in days (default: 30)")
    parser.add_argument("--beat",     default=None,
                        help="Generate report for a single beat only")
    parser.add_argument("--out-dir",  default="outputs/reports")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    anchor    = date.fromisoformat(args.date) if args.date else date.today()
    date_to   = anchor.isoformat()
    date_from = (anchor - timedelta(days=args.days)).isoformat()
    log.info(f"Reporting window: {date_from}  →  {date_to}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        conn = _connect(cfg)
    except Exception as e:
        log.error(f"Cannot connect to DB: {e}")
        sys.exit(1)

    beats = _fetch_beats(conn, date_from, date_to, args.beat)
    if not beats:
        log.warning("No alerts found for the given window/beat. Nothing to report.")
        conn.close()
        return

    log.info(f"Generating reports for {len(beats)} beat(s): {beats}")

    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        log.error("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    stamp   = anchor.strftime("%Y%m%d")
    out_pdf = out_dir / f"beat_report_{stamp}.pdf"

    with PdfPages(str(out_pdf)) as pdf:
        for beat_name in beats:
            alerts = _fetch_beat_alerts(conn, beat_name, date_from, date_to)
            if not alerts:
                continue
            log.info(f"  {beat_name}: {len(alerts)} alerts, "
                     f"{sum(a['area_ha'] or 0 for a in alerts):.1f} ha")
            fig = _make_beat_figure(beat_name, alerts, date_from, date_to)
            pdf.savefig(fig, dpi=150, facecolor=fig.get_facecolor())
            import matplotlib.pyplot as plt
            plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"Van Suraksha Beat Report {stamp}"
        d["Author"]  = "Van Suraksha Pipeline"
        d["Subject"] = "Deforestation Alert Summary by Forest Beat"

    conn.close()
    log.info(f"✅ Report saved: {out_pdf}")
    return str(out_pdf)


if __name__ == "__main__":
    main()
