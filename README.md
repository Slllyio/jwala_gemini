> **⚠️ This repository has been consolidated into [Slllyio/vanagni](https://github.com/Slllyio/vanagni) and is no longer maintained here.**
>
> This repository's only commit (`29d3e2c`) is the root commit of `vanagni`'s history, so everything here is already
> present there, together with all later work. See
> [`docs/CONSOLIDATION.md`](https://github.com/Slllyio/vanagni/blob/main/docs/CONSOLIDATION.md) in that repository.
> This repository will be archived.

# Prithvi Forest-to-Agriculture Change Detection Pipeline

> **Forest change detection and future risk prediction for Fatehgarh Sahib, Punjab, India**
> Using the **IBM/NASA Prithvi-100M** geospatial foundation model + HLS satellite imagery

---

## 🏗️ Architecture Overview

```
GEE (HLS + Hansen GFC)
        ↓
Preprocessing (224×224 patches)
        ↓
┌─────────────────────────────────────────┐
│         Prithvi-100M Encoder            │
│    (ViT, pre-trained on HLS data)       │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Change Head          Prediction Head
(UperNet/FPN)        (ConvLSTM)
    │                     │
    ▼                     ▼
Change Mask          Risk Heatmap
(Forest Loss)        (Future P(change))
    │                     │
    └──────────┬──────────┘
               ▼
     Folium Dashboard (HTML)
```

## 📁 Project Structure

```
prithvi-forest-change/
├── config.yaml                    # All parameters
├── requirements.txt               # Python dependencies
├── scripts/
│   └── run_pipeline.py            # End-to-end pipeline runner
├── src/
│   ├── data/
│   │   ├── gee_fetch.py           # Google Earth Engine data export
│   │   └── preprocess.py          # Tiling, normalization, train/val/test split
│   ├── model/
│   │   ├── prithvi_mae.py         # Prithvi-100M architecture (ViT)
│   │   ├── prithvi_wrapper.py     # HuggingFace model loader
│   │   ├── change_head.py         # FPN segmentation head + Dice+CE loss
│   │   ├── prediction_head.py     # ConvLSTM temporal prediction head
│   │   └── full_model.py          # Combined model (detect + predict modes)
│   ├── train/
│   │   ├── dataset.py             # ForestChangeDataset + TemporalPredictionDataset
│   │   ├── train.py               # Training loop (AMP, TensorBoard, checkpoints)
│   │   └── metrics.py             # IoU, F1, Precision, Recall
│   ├── inference/
│   │   ├── detect.py              # Sliding-window change detection
│   │   └── predict.py             # Risk map generation
│   └── viz/
│       └── dashboard.py           # Folium interactive HTML dashboard
├── data/
│   ├── raw/                       # Downloaded HLS GeoTIFFs (hls_YYYY.tif)
│   ├── labels/                    # Hansen GFC loss masks (loss_YYYY.tif)
│   └── processed/                 # Tiled patches (.npy) + manifests (.csv)
└── outputs/
    ├── checkpoints/               # Saved model weights
    ├── change_map_*.tif           # Detection outputs
    ├── risk_map*.tif              # Prediction outputs
    └── dashboard.html             # Interactive map
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install PyTorch with CUDA (CUDA 12.1 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Authenticate Earth Engine

```bash
earthengine authenticate
```
Your GEE account (`ee-akshayr1`) must have access to `projects/ee-akshayr1/assets/fatehgarh`.

### 3. Fetch Data from GEE

```bash
# Quick test (downloads ~1km² patch directly)
python src/data/gee_fetch.py --config config.yaml --test

# Full export (submits tasks to GEE → downloads from Google Drive)
python src/data/gee_fetch.py --config config.yaml --project ee-akshayr1
```

> After full export: download files from Google Drive `prithvi_fatehgarh/` folder to `data/raw/` (HLS) and `data/labels/` (Hansen loss masks).

### 4. Preprocess

```bash
python src/data/preprocess.py --config config.yaml
```

Outputs: tiled `.npy` patches + `train.csv`, `val.csv`, `test.csv` in `data/processed/`

### 5. Train Change Detection

```bash
python src/train/train.py --config config.yaml --task detect
```

With your local GPU (CUDA), each epoch ~5-15 min depending on AOI size.
Monitor training:
```bash
tensorboard --logdir outputs/tensorboard
```

### 6. Train Future Prediction

```bash
python src/train/train.py --config config.yaml --task predict
```

### 7. Run Inference

```bash
# Change detection
python src/inference/detect.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_detect.pth \
    --output outputs/change_map_2022_2023.tif

# Future risk prediction
python src/inference/predict.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_predict.pth \
    --output outputs/risk_map_2025.tif
```

### 8. Generate Dashboard

```bash
python src/viz/dashboard.py --config config.yaml
# Open: outputs/dashboard.html
```

### Or: Run Everything at Once

```bash
python scripts/run_pipeline.py --config config.yaml --steps all
```

---

## 🧪 Verify Model (No Data Required)

```bash
python src/model/full_model.py --config config.yaml
```

Expected output:
```
[DETECT] Output: torch.Size([1, 2, 224, 224])  ✅
[PREDICT] Output: torch.Size([1, 1, 224, 224])   ✅
Total parameters: ~100,000,000
```

---

## ⚙️ Configuration

Key parameters in `config.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `gee` | `aoi_asset` | GEE asset path for AOI |
| `gee` | `date_range` | Years to fetch imagery |
| `model` | `num_frames` | Temporal frames per input (3) |
| `model` | `freeze_encoder` | Freeze Prithvi weights (recommended: false with GPU) |
| `training` | `batch_size` | Increase if VRAM > 16GB |
| `training` | `amp` | Mixed precision (keeps GPU fast) |
| `prediction` | `num_time_steps` | History length for prediction |

---

## 📊 Expected Results

After training on Hansen GFC labels for Fatehgarh Sahib (2018–2023):

| Metric | Expected Range |
|--------|----------------|
| Change Detection IoU | 0.45 – 0.70 |
| Change Detection F1 | 0.60 – 0.80 |
| Risk Prediction AUC | 0.70 – 0.85 |

> Note: Fatehgarh Sahib is primarily agricultural land. Forest loss signals may be subtle — focus on tree-lined field boundaries and small woodland patches.

---

## 🔗 References

- [IBM/NASA Prithvi-100M — HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
- [NASA-IMPACT HLS Foundation OS](https://github.com/NASA-IMPACT/hls-foundation-os)
- [Hansen Global Forest Change Dataset](https://glad.earthengine.app/view/global-forest-change)
- [HLS (Harmonized Landsat Sentinel-2)](https://hls.gsfc.nasa.gov/)
