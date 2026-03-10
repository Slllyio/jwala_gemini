# eNetra — DW-Delta Alert Validation & ML Pipeline
## Master Implementation Plan  |  v4  |  2026-02-22

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GOOGLE EARTH ENGINE                                  │
│                                                                              │
│  Dynamic World V1 (Sentinel-2, 10m)                                          │
│  ┌─────────────────────┐   ┌─────────────────────┐                          │
│  │  Dec year_a median  │   │  Dec year_b median  │                          │
│  │  8-band prob image  │   │  8-band prob image  │                          │
│  │  + cloud_fraction   │   │  + cloud_fraction   │                          │
│  └──────────┬──────────┘   └──────────┬──────────┘                          │
│             └──────────── Δ ──────────┘                                      │
│                           │                                                  │
│              10-band delta TIF  (per Forest Range, 10m)                      │
│              Bands 1-8: prob deltas    Band 9: cloud_before                  │
│                                        Band 10: cloud_after                  │
│                                                                              │
│  Also exported:  dw_gt_{year}_{class}.tif  (binary change rasters)           │
│                  Classes: deforestation (trees→bare), encroachment (trees→crops) │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │  Download from Drive
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL / PYTHON                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ALERT SIMULATION  (ALERT_GUNA.ipynb)                               │    │
│  │  3-day sliding window on DW composites across full year             │    │
│  │  Output: alert_{year}-MM-DD.geojson  (Source field = DW class name)│    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                  │                                           │
│            ┌─────────────────────┼──────────────────────┐                   │
│            ▼                     ▼                        ▼                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐     │
│  │ validate_alerts  │  │  plot_alert_     │  │  build_feature_stack  │     │
│  │    .py           │  │  dynamics.py     │  │        .py            │     │
│  │                  │  │                  │  │                       │     │
│  │ IoU match alerts │  │ 10-band delta    │  │ 10-band delta +       │     │
│  │ to GT rasters    │  │ raster + alerts  │  │ neighborhood stats +  │     │
│  │ Source vs GT     │  │ per range,       │  │ ancillary → ~40 feat  │     │
│  │ class_match flag │  │ cloud-hatched    │  │ per pixel             │     │
│  └──────────────────┘  └──────────────────┘  └────────────┬──────────┘     │
│                                                            │                 │
│                                                            ▼                 │
│                                              ┌─────────────────────────┐    │
│                                              │  train_lgbm.py  (local) │    │
│                                              │   OR                    │    │
│                                              │  train_gee_rf.py (GEE)  │    │
│                                              │                         │    │
│                                              │  Labels: pixels sampled │    │
│                                              │  from GT binary rasters │    │
│                                              │  0=bg  1=defor  2=encr  │    │
│                                              └────────────┬────────────┘    │
│                                                           │                  │
│                                                           ▼                  │
│                                              ┌─────────────────────────┐    │
│                                              │   predict_guna.py       │    │
│                                              │   Classified raster     │    │
│                                              │   + overlay on viz      │    │
│                                              └─────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**No Prithvi. No GPU required. No vectorisation required.**
The model is a LightGBM or GEE-native Random Forest, trained on pixel features
extracted from the 10-band DW delta raster.

---

## 2. Class Schema

### 2a. Alert Source field (ALERT_GUNA.ipynb output)

The notebook emits one GeoJSON per analysis window. Every feature MUST have a
`Source` attribute whose value is the DW band name describing what changed:

| `Source` value   | What happened at this pixel              | Change narrative          |
|------------------|------------------------------------------|---------------------------|
| `Trees`          | Tree probability dropped ≥ 0.5           | Origin: defor or encr     |
| `Bare`           | Bare probability rose ≥ 0.5             | Destination: deforestation|
| `Crops`          | Crop probability rose ≥ 0.5             | Destination: encroachment |
| `Shrub and Scrub`| Shrub probability changed               | Open dry-forest clearing  |
| `Built`          | Built probability rose                  | Settlement expansion      |
| `Grass`          | Grass probability changed               | Usually seasonal noise     |

**Reading combined alerts at same pixel:**
```
Trees alert  +  Bare alert   at same location  →  deforestation
Trees alert  +  Crops alert  at same location  →  encroachment
Trees alert  +  Built alert  at same location  →  settlement (3rd class)
Only Grass alert                               →  seasonal FP
Any alert    +  cloud_before or cloud_after > 0.4  →  suspect, needs review
```

### 2b. Ground Truth (GEE Dec→Dec annual)

| GT raster name         | Pixel value 1 = change confirmed |
|------------------------|----------------------------------|
| `dw_gt_{year}_deforestation_{range}.tif` | trees/shrub → bare  |
| `dw_gt_{year}_encroachment_{range}.tif`  | trees/shrub → crops |

No vectorisation needed. GT rasters overlay directly.

### 2c. Source ↔ GT cross-reference (validate_alerts.py)

```
Source=Trees         →  consistent with BOTH GT classes
Source=Bare          →  consistent with deforestation GT
Source=Crops         →  consistent with encroachment GT
Source=Shrub and Scrub → consistent with BOTH GT classes
Source=Built or Grass  → neither class  →  class_match=False always
```

---

## 3. The 10-Band Raster (core data product)

All ML and visualisation work from the same 10-band delta TIF.
Size per range: ~200–300 sq km at 10m ≈ ~200–500 MB.

| Band index | Band name        | Range    | Meaning                          |
|-----------|-----------------|----------|----------------------------------|
| 0         | `water`         | -1 → +1  | Δ water probability              |
| 1         | `trees`         | -1 → +1  | Δ tree probability  ← KEY       |
| 2         | `grass`         | -1 → +1  | Δ grass probability              |
| 3         | `flooded_veg`   | -1 → +1  | Δ flooded veg probability        |
| 4         | `crops`         | -1 → +1  | Δ crop probability  ← KEY       |
| 5         | `shrub_and_scrub`| -1 → +1 | Δ shrub probability              |
| 6         | `built`         | -1 → +1  | Δ built probability              |
| 7         | `bare`          | -1 → +1  | Δ bare probability  ← KEY       |
| 8         | `cloud_before`  | 0 → 1    | Cloud fraction Dec year_a        |
| 9         | `cloud_after`   | 0 → 1    | Cloud fraction Dec year_b        |

**Reliability rule:** A pixel where band 8 OR band 9 > 0.4 has fewer than 60%
cloud-free December observations → delta is unreliable → mask in classifier.

---

## 4. Forest Range Split

AOI: `projects/van-suraksha-alert/assets/gunafinal`
Full division: 2144 sq km — too large for one readable figure or fast export.

Guna Division Forest Ranges (extracted from Beat_Name prefix in asset):
```
Aron  |  Bamori  |  Binaganj  |  Fatehgarh  |  Isagarh
Maksudangarh  |  North_Guna  |  Raghogarh  |  South_Guna
```

Each range gets its own:
- Export tasks (3 GeoTIFs: dec_a, dec_b, delta)
- Visualization figure (one PNG per range)
- Per-alert deep-dive panels (optional)
- Feature stack raster
- Classifier prediction raster

**Ranges of highest interest for encroachment:**
`North_Guna`, `Raghogarh` — adjacent to Guna town agricultural expansion.

---

## 5. Script Inventory

### ✅ Already built

| Script | Purpose |
|--------|---------|
| `scripts/export_dw_annual_ground_truth.py` | GT binary rasters for 2019–2025 (7 GEE tasks) |
| `scripts/export_dw_prob_rasters.py`        | 10-band delta TIFs per range (3 GEE tasks × 9 ranges = 27 tasks) |
| `scripts/validate_alerts.py`              | IoU match alerts→GT; Source field used directly; class_match flag |
| `scripts/plot_alert_dynamics.py`          | 10-panel figure per range; cloud hatch; per-alert deep-dive |

### 🔲 To build (next sessions)

| Script | Purpose | Complexity |
|--------|---------|-----------|
| `scripts/build_feature_stack.py` | 10 delta bands + neighborhood stats + ancillary → ~40-feature raster per range | Medium |
| `scripts/train_lgbm.py`          | LightGBM 3-class classifier (bg/defor/encr) with SHAP; pixel samples from GT rasters | Medium |
| `scripts/train_gee_rf.py`        | Alternative: GEE-native `smileRandomForest` — no local compute, runs at GEE scale | Medium |
| `scripts/predict_guna.py`        | Apply trained classifier to each range's feature stack; output classified raster | Low |
| `scripts/run_simulation_year.py` | Wrapper to run ALERT_GUNA logic for every 3-day window in a year; saves daily GeoJSONs | Medium |

---

## 6. Step-by-Step Execution

### Step 1 — GEE exports  (already submitted ✅)
```
GEE Tasks page: https://code.earthengine.google.com/tasks
Monitor until all tasks show COMPLETED.

GT binary rasters     → Drive: prithvi_guna_ground_truth/
Prob delta rasters    → Drive: prithvi_guna_prob_viz/
```

### Step 2 — Download and organise
```
data/
  ground_truth/
    North_Guna/
      dw_prob_delta_2024_2025_North_Guna.tif   ← 10-band
      dw_gt_2025_deforestation_North_Guna.tif
      dw_gt_2025_encroachment_North_Guna.tif
    Raghogarh/
      dw_prob_delta_2024_2025_Raghogarh.tif
      ...
  alerts/
    alert_2025-01-03.geojson     ← from ALERT_GUNA simulation
    alert_2025-01-06.geojson
    ...
  validation/                    ← created by validate_alerts.py
  viz/                           ← created by plot_alert_dynamics.py
```

### Step 3 — 2025 Alert Simulation
```
Open ALERT_GUNA.ipynb
Run for each 3-day window throughout 2025 (Jan 1 – Dec 31)
Save each result: data/alerts/alert_2025-MM-DD.geojson
Verify `Source` field is present in every feature.
```

### Step 4 — Validate alerts
```bash
python scripts/validate_alerts.py --year 2025

# Output:
#   data/validation/alert_validation_2025.geojson   (Source + matched + class_match)
#   data/validation/report_2025.txt                 (metrics)
```

Report structure:
```
Precision: matched / total_alerts           (how many alert polygons hit a real change)
Recall:    GT_caught / total_GT_changes     (how many real changes had at least one alert)

Per Source breakdown:
  Trees          : 47 alerts   38 matched (81%)   class_match: 38 (81%)
  Bare           : 23 alerts   19 matched (83%)   class_match: 19 (83%)
  Crops          : 18 alerts   15 matched (83%)   class_match: 15 (83%)
  Shrub and Scrub: 12 alerts    7 matched (58%)   class_match:  7 (58%)
  Built          :  3 alerts    2 matched (67%)   class_match:  0 ( 0%)  ← settlement
  Grass          :  8 alerts    1 matched (13%)   class_match:  0 ( 0%)  ← seasonal noise
```

### Step 5 — Visualise dynamics (per range)
```bash
# All ranges
python scripts/plot_alert_dynamics.py --year 2025

# Single range, filtered to deforestation signals only
python scripts/plot_alert_dynamics.py --year 2025 --range North_Guna --source Trees,Bare

# Per-alert deep-dive panels for all North_Guna alerts
python scripts/plot_alert_dynamics.py --year 2025 --range North_Guna --per-alert

# Output: data/viz/alert_dynamics_2025_{range}.png  (one per range)
#         data/viz/per_alert_2025_{range}/           (one per alert, if --per-alert)
```

**Reading each figure:**
```
RGB panel (top-left half):
  Deep red     → trees lost, nothing yet planted  (deforestation)
  Warm orange  → trees lost AND bare exposed      (deforestation)
  Bright green → crop probability gained           (encroachment destination)

GT raster overlay:
  Semi-transparent red   → confirmed deforestation polygon
  Semi-transparent orange → confirmed encroachment polygon

Alert outlines (by Source colour):
  Green  = Trees alert  |  Yellow = Crops  |  Orange = Bare
  Brown  = Shrub        |  Red    = Built  |  Lime   = Grass (dashed)
  White dots            = False positives (no GT match)
  Yellow dashes         = Missed GT changes

Cloud panel (bottom-right):
  YlOrRd scale = max(cloud_before, cloud_after) per pixel
  Grey hatch   = unreliable zone (>40% cloud fraction)
  Same alert outlines: assess if alert falls in hatch → suspect FP
```

### Step 6 — Feature Engineering (NEXT TO BUILD)
```bash
python scripts/build_feature_stack.py --range North_Guna --year-a 2024 --year-b 2025
```

Feature stack per pixel (~40 features):
```
Raw deltas (8):       bands 0-7 from delta TIF
Cloud flags (2):      bands 8-9 from delta TIF
Neighborhood mean 3×3 (4):  trees, shrub, crops, bare deltas
Neighborhood mean 7×7 (4):  same
Neighborhood std  5×5 (4):  same
Edge sharpness  Sobel (3):  trees, crops, bare (sharp edge = cutting event)
Pre-event tree prob   (1):  trees_prob_dec{year_a}  (was it actually forested?)
Pre-event shrub prob  (1):  shrub_prob_dec{year_a}
Slope           SRTM  (1):  from existing friction layer
Distance to road      (1):  from existing friction layer
Cloud reliable flag   (1):  0/1 binary (cloud < 0.4 in both epochs)
```

### Step 7 — Train Classifier

**Option A: LightGBM (recommended — SHAP, fast, no GPU)**
```bash
python scripts/train_lgbm.py --range North_Guna,Raghogarh --year 2025

# Samples pixels from GT binary rasters:
#   class 0 = background  (GT raster = 0 in both defor and encr)
#   class 1 = deforestation (gt_defor raster = 1)
#   class 2 = encroachment  (gt_encr raster = 1)
# cloud_reliable_flag = 0 → excluded from training

# Output: models/lgbm_v1.pkl  +  models/feature_importance_shap.png
```

**Option B: GEE-native Random Forest (no download needed)**
```bash
python scripts/train_gee_rf.py --year 2025

# Runs entirely inside GEE:
#   ee.Classifier.smileRandomForest(300)
#   .train(features=gt_samples, classProperty="class", inputProperties=feature_names)
# Output: GEE asset — classified_guna_2025
```

### Step 8 — Predict and Overlay
```bash
python scripts/predict_guna.py --range North_Guna --year 2025

# Output:
#   data/predictions/predicted_2025_North_Guna.tif
#   data/viz/prediction_overlay_2025_North_Guna.png
```

---

## 7. Future: Google Vertex AI Geospatial Embeddings (Phase 2)

If LightGBM recall is insufficient (< 60%), upgrade the feature extraction:

```python
# GEE call to Google's Geospatial Foundation Model (if GCP access confirmed)
embedding_model = ee.Model.fromVertexAi(
    endpoint="projects/earthengine-public/models/geospatial-fm",
    inputTileSize=[64, 64],
    outputBands={"embedding": {"type": ee.PixelType.float(), "dimensions": 512}}
)
embeddings = embedding_model.predictImage(dw_composite_10band)
# → 512-dim embedding per 64×64 pixel chip
# Add thin 3-layer MLP head trained on GT raster labels
```

This is the "Google Earth embeddings" path — DW probability delta (10 bands)
feeds the GFM which generates semantically rich 512-dim embeddings.
A linear probe on these beats LightGBM on area coverage and shape precision.

---

## 8. Summary of Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Label source | Dynamic World V1 Dec→Dec | Hansen blind to dry-deciduous forest |
| Feature space | 10-band DW delta TIF | DW probs ARE Google Earth embeddings |
| Cloud handling | Bands 9-10 in same TIF | Flag unreliable pixels at source |
| AOI splitting | Per Forest Range (~250 sq km) | Full 2144 sq km = unreadable |
| Alert class | `Source` field from notebook | Alerts already use DW taxonomy |
| GT format | Binary raster | No vectorisation needed |
| Model | LightGBM or GEE RF | No GPU, SHAP explainability, fast |
| Backbone | None (Prithvi killed) | 10-band DW delta is self-sufficient |
