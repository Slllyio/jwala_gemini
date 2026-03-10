"""
Flywheel Retraining — Close the Feedback Loop
================================================

Queries human labels from flywheel_labels + alert geometries from
alerts_log, combines them with existing training data, and retrains
the LightGBM alert filter.

Usage:
    python scripts/retrain_with_labels.py --config config.yaml
    python scripts/retrain_with_labels.py --config config.yaml --min-labels 20 --version flywheel_v1

Flow:
    1. Query confirmed + false_positive labels from PostGIS
    2. Extract features from labelled alert rasters
    3. Merge with original training data (if available)
    4. Train LightGBM with weighted samples
    5. Evaluate and save training report
    6. Auto-register new model version
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import yaml
import psycopg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_labelled_alerts(conn, min_labels: int = 10):
    """Query labelled alerts from the flywheel tables."""
    cur = conn.cursor()
    cur.execute("""
        SELECT al.id, al.change_type, al.area_ha,
               COALESCE(al.stacked_score, al.confidence) AS score,
               al.model_version, al.centroid_lat, al.centroid_lon,
               al.mean_delta_trees, al.mean_delta_crops, al.mean_delta_built,
               al.mean_delta_bare, al.mean_delta_grass, al.mean_delta_water,
               fl.label, fl.labeler, fl.notes
        FROM flywheel_labels fl
        JOIN alerts_log al ON fl.alert_id = al.id
        WHERE fl.label IN ('confirmed', 'false_positive')
        ORDER BY fl.labeled_at DESC;
    """)

    rows = cur.fetchall()
    if len(rows) < min_labels:
        log.warning(f"Only {len(rows)} labels found (minimum: {min_labels}). "
                    f"Need more labels before retraining.")
        return None

    columns = [
        "id", "change_type", "area_ha", "score", "model_version",
        "centroid_lat", "centroid_lon",
        "mean_delta_trees", "mean_delta_crops", "mean_delta_built",
        "mean_delta_bare", "mean_delta_grass", "mean_delta_water",
        "label", "labeler", "notes"
    ]
    df = pd.DataFrame(rows, columns=columns)
    log.info(f"Loaded {len(df)} labelled alerts "
             f"({(df['label'] == 'confirmed').sum()} confirmed, "
             f"{(df['label'] == 'false_positive').sum()} false_positive)")
    return df


def build_training_features(df: pd.DataFrame):
    """Build feature matrix from labelled alerts."""
    # Feature columns — use available delta/score/area features
    feature_cols = [
        "score", "area_ha",
        "mean_delta_trees", "mean_delta_crops", "mean_delta_built",
        "mean_delta_bare", "mean_delta_grass", "mean_delta_water",
        "centroid_lat", "centroid_lon",
    ]

    X = df[feature_cols].fillna(0).values
    y = (df["label"] == "confirmed").astype(int).values

    log.info(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    log.info(f"Label distribution: {y.sum()} positive, {(1-y).sum()} negative")

    return X, y, feature_cols


def load_existing_training_data(data_dir: str = "outputs/alert_filter"):
    """Load existing training data from previous experiments (if available)."""
    train_path = Path(data_dir) / "training_data.npz"
    if not train_path.exists():
        log.info("No existing training data found, using labels only.")
        return None, None

    data = np.load(train_path)
    X_existing = data["X"]
    y_existing = data["y"]
    log.info(f"Loaded existing training data: {X_existing.shape}")
    return X_existing, y_existing


def train_model(X, y, feature_names, output_dir: str, version: str):
    """Train LightGBM model on combined data."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed. Install with: pip install lightgbm")
        return None

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        balanced_accuracy_score, average_precision_score,
        f1_score, precision_score, recall_score, classification_report
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Class weights for imbalanced data
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 5,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=min(5, min(n_pos, n_neg)), shuffle=True, random_state=42)
    model = lgb.LGBMClassifier(**params)

    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="balanced_accuracy")
    log.info(f"CV Balanced Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Train final model on all data
    model.fit(X, y)

    # Predictions on training data (for report)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "average_precision": float(average_precision_score(y, y_prob)),
        "f1_score": float(f1_score(y, y_pred)),
        "precision_at_threshold": float(precision_score(y, y_pred)),
        "recall_at_threshold": float(recall_score(y, y_pred)),
        "cv_balanced_accuracy_mean": float(cv_scores.mean()),
        "cv_balanced_accuracy_std": float(cv_scores.std()),
        "n_samples": len(y),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
    }

    # Save model
    model_path = out / "alert_filter.lgbm"
    model.booster_.save_model(str(model_path))
    log.info(f"Model saved: {model_path}")

    # Save training report
    report = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": "flywheel_labels",
        "metrics": metrics,
        "feature_config": {
            "feature_names": feature_names,
            "n_features": len(feature_names),
        },
        "params": params,
        "classification_report": classification_report(y, y_pred, output_dict=True),
    }
    report_path = out / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Report saved: {report_path}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info(f"\nFeature Importance:\n{importance.to_string(index=False)}")

    return report


def auto_register(output_dir: str, version: str, config_path: str):
    """Auto-register the new model version."""
    try:
        from scripts.model_store import register
        register(output_dir, version, notes="Flywheel retrained", config_path=config_path)
    except Exception as e:
        log.warning(f"Auto-register failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Van Suraksha -- Flywheel Retraining")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--min-labels", type=int, default=10,
                        help="Minimum labels required before retraining")
    parser.add_argument("--version", default=None,
                        help="Version name for new model (default: flywheel_YYYYMMDD)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/alert_filter_flywheel)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    db = cfg.get("database", {})
    conn = psycopg.connect(
        host=db.get("host", "localhost"),
        port=db.get("port", 5432),
        dbname=db.get("dbname", "gis_projects"),
        user=db.get("user", "postgres"),
        password=os.environ.get("VS_DB_PASSWORD", db.get("password", "")),
    )

    try:
        # 1. Query labels
        df = get_labelled_alerts(conn, min_labels=args.min_labels)
        if df is None:
            log.info("Exiting — insufficient labels for retraining.")
            return

        # 2. Build features
        X, y, feature_names = build_training_features(df)

        # 3. Try to merge with existing data
        X_existing, y_existing = load_existing_training_data()
        if X_existing is not None:
            # Align feature dimensions (pad or truncate)
            if X_existing.shape[1] != X.shape[1]:
                log.warning(f"Feature dimension mismatch: existing={X_existing.shape[1]}, "
                            f"new={X.shape[1]}. Using labels only.")
            else:
                X = np.vstack([X_existing, X])
                y = np.concatenate([y_existing, y])
                log.info(f"Combined: {X.shape[0]} samples (existing + labels)")

        # 4. Train
        version = args.version or f"flywheel_{datetime.now().strftime('%Y%m%d')}"
        output_dir = args.output_dir or f"outputs/alert_filter_{version}"

        report = train_model(X, y, feature_names, output_dir, version)
        if report is None:
            return

        # 5. Auto-register
        auto_register(output_dir, version, args.config)

        log.info(f"\n[DONE] Flywheel retraining complete: {version}")
        log.info(f"  Balanced Accuracy: {report['metrics']['balanced_accuracy']:.4f}")
        log.info(f"  F1 Score: {report['metrics']['f1_score']:.4f}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
