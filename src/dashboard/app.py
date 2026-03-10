"""
Van Suraksha Interactive Dashboard
====================================

Streamlit app for live alert exploration, label submission,
and pipeline monitoring. Connects to PostGIS.

Run:
    streamlit run src/dashboard/app.py

Docker:
    docker compose run -p 8501:8501 app streamlit run src/dashboard/app.py
"""

import os
import json
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import psycopg
import yaml

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Van Suraksha Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_PATH = os.environ.get("VS_CONFIG", "config.yaml")
try:
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    _db = _cfg.get("database", {})
except FileNotFoundError:
    _db = {}

DB_DSN = (
    f"host={_db.get('host', 'localhost')} "
    f"port={_db.get('port', 5432)} "
    f"dbname={_db.get('dbname', 'gis_projects')} "
    f"user={_db.get('user', 'postgres')} "
    f"password={os.environ.get('VS_DB_PASSWORD', _db.get('password', ''))}"
)


@st.cache_resource
def get_conn():
    return psycopg.connect(DB_DSN, autocommit=True)


def query_df(sql, params=None):
    conn = get_conn()
    return pd.read_sql(sql, conn, params=params)


# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.title("Van Suraksha")
st.sidebar.markdown("*Guna Forest Division Alert System*")

# Model version filter
versions = query_df(
    "SELECT DISTINCT model_version FROM alerts_log ORDER BY model_version"
)["model_version"].tolist()
selected_version = st.sidebar.selectbox(
    "Model Version", ["All"] + versions
)

# Change type filter
change_types = query_df(
    "SELECT DISTINCT change_type FROM alerts_log WHERE change_type IS NOT NULL ORDER BY change_type"
)["change_type"].tolist()
selected_type = st.sidebar.multiselect("Change Type", change_types, default=change_types)

# Confidence filter
min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)

# Area filter
min_area = st.sidebar.slider("Min Area (ha)", 0.0, 50.0, 0.0, 0.5)

# Date filter
date_range = st.sidebar.date_input(
    "Detection Date Range",
    value=(date(2023, 1, 1), date.today()),
    min_value=date(2018, 1, 1),
    max_value=date.today(),
)

# ── Build Query ──────────────────────────────────────────────────────────────
conditions = ["1=1"]
params = {}

if selected_version != "All":
    conditions.append("model_version = %(version)s")
    params["version"] = selected_version

if selected_type:
    conditions.append("change_type = ANY(%(types)s)")
    params["types"] = selected_type

conditions.append("COALESCE(stacked_score, confidence) >= %(min_conf)s")
params["min_conf"] = min_conf

conditions.append("area_ha >= %(min_area)s")
params["min_area"] = min_area

if len(date_range) == 2:
    conditions.append("detection_date >= %(date_from)s")
    conditions.append("detection_date <= %(date_to)s")
    params["date_from"] = str(date_range[0])
    params["date_to"] = str(date_range[1])

where = " AND ".join(conditions)

# ── Main Content ─────────────────────────────────────────────────────────────
tab_map, tab_table, tab_stats, tab_label, tab_runs = st.tabs(
    ["Map", "Alert Table", "Statistics", "Label Alerts", "Pipeline Runs"]
)

# --- Tab 1: Map ---
with tab_map:
    st.subheader("Alert Map")
    alerts_geo = query_df(
        f"""SELECT id, change_type, area_ha,
                   COALESCE(stacked_score, confidence) AS conf,
                   centroid_lat AS lat, centroid_lon AS lon,
                   detection_date, model_version
            FROM alerts_log WHERE {where}
            AND centroid_lat IS NOT NULL AND centroid_lon IS NOT NULL
            ORDER BY COALESCE(stacked_score, confidence) DESC NULLS LAST
            LIMIT 2000""",
        params,
    )

    if alerts_geo.empty:
        st.info("No alerts match your filters.")
    else:
        st.map(alerts_geo, latitude="lat", longitude="lon", size=20)
        st.caption(f"Showing {len(alerts_geo)} alerts")

# --- Tab 2: Table ---
with tab_table:
    st.subheader("Alert Details")
    alerts_df = query_df(
        f"""SELECT id, change_type, area_ha,
                   COALESCE(stacked_score, confidence) AS confidence,
                   detection_date, model_version, sub_range,
                   mean_delta_trees, mean_delta_crops, mean_delta_built
            FROM alerts_log WHERE {where}
            ORDER BY COALESCE(stacked_score, confidence) DESC NULLS LAST
            LIMIT 500""",
        params,
    )
    st.dataframe(alerts_df, use_container_width=True, height=500)
    st.download_button(
        "Download CSV", alerts_df.to_csv(index=False), "alerts.csv", "text/csv"
    )

# --- Tab 3: Statistics ---
with tab_stats:
    st.subheader("Alert Statistics")

    col1, col2, col3 = st.columns(3)
    summary = query_df(
        f"SELECT COUNT(*), COALESCE(SUM(area_ha),0), COALESCE(AVG(COALESCE(stacked_score, confidence)),0) FROM alerts_log WHERE {where}",
        params,
    )
    if not summary.empty:
        col1.metric("Total Alerts", f"{summary.iloc[0, 0]:,}")
        col2.metric("Total Area", f"{summary.iloc[0, 1]:.1f} ha")
        col3.metric("Avg Confidence", f"{summary.iloc[0, 2]:.3f}")

    # By change type
    st.markdown("#### By Change Type")
    by_type = query_df(
        f"""SELECT change_type, COUNT(*) AS count,
                   ROUND(AVG(area_ha)::numeric, 2) AS avg_area,
                   ROUND(SUM(area_ha)::numeric, 1) AS total_area
            FROM alerts_log WHERE {where}
            GROUP BY change_type ORDER BY count DESC""",
        params,
    )
    if not by_type.empty:
        st.bar_chart(by_type.set_index("change_type")["count"])
        st.dataframe(by_type)

    # By version
    st.markdown("#### By Model Version")
    by_ver = query_df(
        f"""SELECT model_version, COUNT(*) AS count,
                   ROUND(AVG(COALESCE(stacked_score, confidence))::numeric, 4) AS avg_conf
            FROM alerts_log WHERE {where}
            GROUP BY model_version ORDER BY count DESC""",
        params,
    )
    if not by_ver.empty:
        st.dataframe(by_ver)

# --- Tab 4: Label Alerts ---
with tab_label:
    st.subheader("Label Alerts for Flywheel")
    st.markdown("Submit labels to improve future model performance.")

    alert_id = st.number_input("Alert ID", min_value=1, step=1)
    label_choice = st.selectbox("Label", ["confirmed", "false_positive", "uncertain"])
    labeler_name = st.text_input("Your Name", "dashboard_user")
    label_notes = st.text_area("Notes (optional)")

    if st.button("Submit Label", type="primary"):
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO flywheel_labels (alert_id, label, labeler, notes)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (alert_id) DO UPDATE
                   SET label = EXCLUDED.label, labeler = EXCLUDED.labeler,
                       notes = EXCLUDED.notes, labeled_at = NOW()""",
                [alert_id, label_choice, labeler_name, label_notes],
            )
            st.success(f"Label saved: Alert #{alert_id} = {label_choice}")
        except Exception as e:
            st.error(f"Error: {e}")

    # Label progress
    st.markdown("#### Label Progress")
    labels = query_df(
        "SELECT label, COUNT(*) AS count FROM flywheel_labels GROUP BY label"
    )
    if not labels.empty:
        st.dataframe(labels)
    else:
        st.info("No labels yet. Start labelling!")

# --- Tab 5: Pipeline Runs ---
with tab_runs:
    st.subheader("Recent Pipeline Runs")
    try:
        runs = query_df(
            """SELECT run_id, run_name, status, duration_s,
                      steps_total, steps_ok, steps_failed, created_at
               FROM pipeline_runs
               ORDER BY created_at DESC LIMIT 20"""
        )
        if not runs.empty:
            st.dataframe(runs, use_container_width=True)
        else:
            st.info("No pipeline runs recorded yet.")
    except Exception:
        st.info("Pipeline runs table not yet created.")
