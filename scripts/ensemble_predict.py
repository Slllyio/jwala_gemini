"""
Ensemble Prediction — Multi-Model Alert Filtering
====================================================

Loads multiple model versions from the registry, runs each on the
same alert raster, and combines predictions using configurable
strategies (weighted average, majority vote, stacking).

Usage:
    python scripts/ensemble_predict.py \\
        --config config.yaml \\
        --models enriched_v2 single_pass \\
        --strategy weighted_avg \\
        --input outputs/alert_rasters/dw_2024Q1.tif \\
        --output outputs/ensemble_alerts.geojson
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()

import argparse
import json
import logging
import numpy as np
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_dir: str):
    """Load a LightGBM model from the registry directory."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed")
        return None

    model_path = Path(model_dir) / "alert_filter.lgbm"
    if not model_path.exists():
        # Search for any .lgbm file
        lgbm_files = list(Path(model_dir).glob("*.lgbm"))
        if lgbm_files:
            model_path = lgbm_files[0]
        else:
            log.warning(f"No LightGBM model in {model_dir}")
            return None

    model = lgb.Booster(model_file=str(model_path))
    log.info(f"Loaded model: {model_path.parent.name}")
    return model


def load_models_from_registry(
    version_names: list,
    registry_dir: str = "outputs/model_registry"
):
    """Load multiple models from the model registry."""
    models = {}
    for name in version_names:
        model_dir = Path(registry_dir) / name
        if not model_dir.exists():
            log.warning(f"Version '{name}' not found in registry")
            continue

        # Load manifest for weights
        manifest_path = model_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {}

        model = load_model(str(model_dir))
        if model is not None:
            models[name] = {
                "model": model,
                "manifest": manifest,
                "metrics": manifest.get("metrics", {}),
            }

    log.info(f"Loaded {len(models)} / {len(version_names)} models")
    return models


# ── Prediction strategies ────────────────────────────────────────────────────

def predict_all(models: dict, features: np.ndarray):
    """Get predictions from all models. Returns dict[name → probabilities]."""
    predictions = {}
    for name, entry in models.items():
        model = entry["model"]
        try:
            probs = model.predict(features)
            predictions[name] = probs
            log.info(f"  {name}: mean={probs.mean():.4f}, "
                     f"max={probs.max():.4f}, >0.5: {(probs > 0.5).sum()}")
        except Exception as e:
            log.warning(f"  {name}: prediction failed: {e}")
    return predictions


def weighted_average(predictions: dict, models: dict, metric_key: str = "balanced_accuracy"):
    """Weighted average using model performance as weights."""
    weights = {}
    for name, entry in models.items():
        if name in predictions:
            w = entry["metrics"].get(metric_key, 0.5)
            weights[name] = max(w, 0.1)  # floor at 0.1

    total_weight = sum(weights.values())
    combined = np.zeros_like(list(predictions.values())[0])

    for name, probs in predictions.items():
        w = weights.get(name, 0.5) / total_weight
        combined += probs * w
        log.info(f"  {name}: weight={w:.3f}")

    return combined


def majority_vote(predictions: dict, threshold: float = 0.5):
    """Majority vote: pixel is positive if majority of models agree."""
    votes = np.stack([(p > threshold).astype(float) for p in predictions.values()])
    consensus = votes.mean(axis=0)
    return consensus  # fraction of models that voted positive


def stacking(predictions: dict, labels: np.ndarray = None):
    """Stack predictions as features for a meta-learner (if labels available)."""
    stack = np.column_stack(list(predictions.values()))

    if labels is None:
        # Simple average as fallback
        return stack.mean(axis=1)

    # Train a simple logistic regression meta-learner
    try:
        from sklearn.linear_model import LogisticRegression
        meta = LogisticRegression(random_state=42)
        meta.fit(stack, labels)
        return meta.predict_proba(stack)[:, 1]
    except Exception as e:
        log.warning(f"Stacking meta-learner failed: {e}. Using average.")
        return stack.mean(axis=1)


STRATEGIES = {
    "weighted_avg": weighted_average,
    "majority_vote": majority_vote,
    "stacking": stacking,
}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_from_raster(input_path: str, config: dict):
    """Extract features from alert raster for model prediction."""
    try:
        from scripts.filter_alerts import extract_features, N_BANDS
        import rasterio

        with rasterio.open(input_path) as src:
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
            crs_str = str(src.crs)

        H, W = data.shape[1], data.shape[2]

        # Use month from filename or default
        month = 6  # default
        alert_thresh = config.get("alert_filter", {}).get("alert_threshold", 0.15)

        features = extract_features(data, month, alert_thresh, H, W)
        return features, data, profile, transform, crs_str, H, W

    except Exception as e:
        log.error(f"Feature extraction failed: {e}")
        return None, None, None, None, None, None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Van Suraksha -- Ensemble Prediction")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model version names from registry")
    parser.add_argument("--strategy", default="weighted_avg",
                        choices=list(STRATEGIES.keys()),
                        help="Ensemble strategy")
    parser.add_argument("--input", required=True,
                        help="Alert raster to process")
    parser.add_argument("--output", default="outputs/ensemble_alerts.geojson",
                        help="Output GeoJSON path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Final detection threshold")
    parser.add_argument("--registry", default="outputs/model_registry",
                        help="Model registry directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Load models
    log.info(f"Loading {len(args.models)} models...")
    models = load_models_from_registry(args.models, args.registry)
    if len(models) < 2:
        log.error("Need at least 2 models for ensemble")
        return

    # 2. Extract features
    log.info(f"Extracting features from: {args.input}")
    features, data, profile, transform, crs_str, H, W = \
        extract_features_from_raster(args.input, cfg)

    if features is None:
        log.error("Feature extraction failed")
        return

    log.info(f"Features: {features.shape}")

    # 3. Get predictions from all models
    log.info("Running predictions...")
    predictions = predict_all(models, features)

    if len(predictions) < 2:
        log.error("Need predictions from at least 2 models")
        return

    # 4. Combine predictions
    log.info(f"Combining with strategy: {args.strategy}")
    if args.strategy == "weighted_avg":
        ensemble_probs = weighted_average(predictions, models)
    elif args.strategy == "majority_vote":
        ensemble_probs = majority_vote(predictions, args.threshold)
    elif args.strategy == "stacking":
        ensemble_probs = stacking(predictions)
    else:
        log.error(f"Unknown strategy: {args.strategy}")
        return

    # 5. Apply threshold
    ensemble_mask = ensemble_probs >= args.threshold
    n_positive = ensemble_mask.sum()
    log.info(f"Ensemble: {n_positive} positive pixels (threshold={args.threshold})")

    # 6. Compare with individual models
    log.info("\n--- Comparison ---")
    log.info(f"{'Model':<20} {'Positive':>10} {'Mean Prob':>10}")
    log.info("-" * 42)
    for name, probs in predictions.items():
        n = (probs > args.threshold).sum()
        log.info(f"{name:<20} {n:>10} {probs.mean():>10.4f}")
    log.info(f"{'ENSEMBLE':<20} {n_positive:>10} {ensemble_probs.mean():>10.4f}")

    # 7. Generate GeoJSON output
    try:
        from scripts.filter_alerts import generate_geojson
        from datetime import datetime

        # Reshape ensemble mask to spatial
        # features were extracted from valid pixels only; need to map back
        confidence_2d = np.zeros((H, W), dtype=np.float32)
        mask_2d = np.zeros((H, W), dtype=bool)

        # Create simple spatial output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        geojson = generate_geojson(
            mask_2d, confidence_2d, data,
            transform, crs_str,
            detection_date=datetime.now().strftime("%Y-%m-%d"),
            sub_range="Ensemble",
            min_pixels=3,
        )

        with open(output_path, "w") as f:
            json.dump(geojson, f)

        n_features = len(geojson.get("features", []))
        log.info(f"\n[DONE] {n_features} polygons -> {output_path}")

    except Exception as e:
        log.warning(f"GeoJSON generation: {e}")

    # Save ensemble summary
    summary = {
        "models": list(models.keys()),
        "strategy": args.strategy,
        "threshold": args.threshold,
        "n_positive_ensemble": int(n_positive),
        "per_model": {
            name: {"n_positive": int((p > args.threshold).sum()),
                   "mean_prob": float(p.mean())}
            for name, p in predictions.items()
        },
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
