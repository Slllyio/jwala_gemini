"""
src/analysis/phenology_model.py
================================
Harmonic regression phenology modelling for Guna Division forests.

Theory
------
We model the expected NDVI for a given day-of-year (DOY) as a 2nd-order
Fourier (harmonic) series:

    NDVI(DOY) = a0
              + a1·cos(2π·DOY/T) + b1·sin(2π·DOY/T)   ← annual cycle
              + a2·cos(4π·DOY/T) + b2·sin(4π·DOY/T)   ← biannual cycle
              + ε

where T = 365.25 (days/year).

This 5-parameter model captures:
  • The broad monsoon peak (annual term)
  • The secondary post-monsoon flush and drought-season trough (biannual term)
  • A mean offset (a0)

The residual standard deviation σ(DOY) quantifies natural variability for
that calendar week.  An anomaly z-score is then:

    z = (observed_NDVI − predicted_NDVI) / σ

A z-score of −2 or below (≡ NDVI is 2σ below expected) is equivalent to
a confidence of ~97.5% that the drop is abnormal — the same significance
level used by RADD-SAR.

Usage
-----
    from src.analysis.phenology_model import PhenologyModel

    # Fit from a list of (date_str, ndvi) pairs
    model = PhenologyModel.fit(records, label="North_Guna/Mar_Ki_Mahu")
    model.save("data/phenology/North_Guna__Mar_Ki_Mahu.json")

    # Score a new observation
    z = model.zscore("2026-02-23", observed_ndvi=0.18)
    # z ≈ -2.3 → strong anomaly → likely felling

    # Load saved model
    model = PhenologyModel.load("data/phenology/North_Guna__Mar_Ki_Mahu.json")
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
_T     = 365.25          # days per year
_MIN_OBS = 12            # minimum observations needed to fit a model
_ZSCORE_ALERT_THRESH = -2.0   # 97.5th percentile anomaly


# ── helpers ──────────────────────────────────────────────────────────────────
def _doy(d: str | date | datetime) -> float:
    """Return fractional day-of-year (1–366) for a date string or object."""
    if isinstance(d, str):
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d"):
            try:
                d = datetime.strptime(d, fmt)
                break
            except ValueError:
                pass
        if isinstance(d, str):
            raise ValueError(f"Cannot parse date: {d!r}")
    if isinstance(d, datetime):
        d = d.date()
    return d.timetuple().tm_yday + (d.month - 1) / 12.0  # slight fraction for month-centre


def _design_matrix(doys: np.ndarray) -> np.ndarray:
    """
    Build the 5-column harmonic design matrix for given DOY array.
    Columns: [1, cos1, sin1, cos2, sin2]
    """
    omega = 2 * math.pi / _T
    X = np.column_stack([
        np.ones(len(doys)),
        np.cos(omega * doys),
        np.sin(omega * doys),
        np.cos(2 * omega * doys),
        np.sin(2 * omega * doys),
    ])
    return X


# ── model dataclass ──────────────────────────────────────────────────────────
@dataclass
class PhenologyModel:
    """
    Fitted harmonic phenology model for one label (beat or division).

    Attributes
    ----------
    label       : identifier string, e.g. "North_Guna/Mar_Ki_Mahu"
    variable    : "ndvi" or "dw_trees"
    coeffs      : list of 5 floats [a0, a1, b1, a2, b2]
    residual_std: float — mean residual σ across all training observations
    residual_doy_std: dict {DOY_bin → σ} — optional per-DOY σ for finer scoring
    n_obs       : number of training observations
    date_range  : [min_date, max_date] of training data
    r2          : coefficient of determination of fit
    """
    label:            str
    variable:         str
    coeffs:           list[float]         # [a0, a1, b1, a2, b2]
    residual_std:     float
    residual_doy_std: dict[str, float]    # DOY_bin (str key for JSON) → σ
    n_obs:            int
    date_range:       list[str]
    r2:               float

    # ────────────────────────────────────────────────────────────────────────
    @classmethod
    def fit(
        cls,
        records: list[dict],
        label:   str = "unknown",
        variable: str = "ndvi",
        doy_bins: int = 12,           # number of DOY bins for local σ (1 per month)
    ) -> "PhenologyModel":
        """
        Fit a 2nd-order harmonic regression to a list of monthly records.

        Parameters
        ----------
        records : list of dicts with keys:
                    'date_label' (YYYY-MM or YYYY-MM-DD)
                    'ndvi_mean'  (or 'dw_mean' etc.)
                    optionally 'ndvi_std'
        label   : identifier string
        variable: 'ndvi' or 'dw_trees'
        """
        val_key = "ndvi_mean" if variable == "ndvi" else "dw_mean"

        # Filter to records with valid values
        clean = [(r["date_label"], r[val_key])
                 for r in records
                 if r.get(val_key) is not None and not math.isnan(float(r[val_key]))]

        if len(clean) < _MIN_OBS:
            raise ValueError(
                f"Too few observations ({len(clean)}) to fit model for {label!r}. "
                f"Need at least {_MIN_OBS}."
            )

        dates, values = zip(*clean)
        doys   = np.array([_doy(d) for d in dates])
        y      = np.array([float(v) for v in values])

        # ── ordinary least-squares fit ────────────────────────────────────
        X      = _design_matrix(doys)
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

        y_hat  = X @ coeffs
        resid  = y - y_hat
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ── global residual σ ─────────────────────────────────────────────
        global_sigma = float(np.std(resid, ddof=min(5, len(resid) - 1)))

        # ── per-DOY-bin residual σ  (12 monthly bins) ────────────────────
        bin_edges = np.linspace(0, 366, doy_bins + 1)
        doy_sigma: dict[str, float] = {}
        for i in range(doy_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask   = (doys >= lo) & (doys < hi)
            if mask.sum() >= 2:
                doy_sigma[str(i)] = float(np.std(resid[mask], ddof=1))
            else:
                doy_sigma[str(i)] = global_sigma   # fallback

        date_range = [str(min(dates)), str(max(dates))]

        return cls(
            label            = label,
            variable         = variable,
            coeffs           = coeffs.tolist(),
            residual_std     = global_sigma,
            residual_doy_std = doy_sigma,
            n_obs            = len(clean),
            date_range       = date_range,
            r2               = r2,
        )

    # ────────────────────────────────────────────────────────────────────────
    def predict(self, date_or_doy: str | float | date) -> float:
        """Return expected NDVI for a given date or DOY."""
        doy = _doy(date_or_doy) if not isinstance(date_or_doy, (int, float)) else float(date_or_doy)
        X   = _design_matrix(np.array([doy]))
        return float((X @ np.array(self.coeffs))[0])

    def sigma(self, date_or_doy: str | float | date, doy_bins: int = 12) -> float:
        """Return local residual σ for a given date."""
        doy = _doy(date_or_doy) if not isinstance(date_or_doy, (int, float)) else float(date_or_doy)
        bin_edges = np.linspace(0, 366, doy_bins + 1)
        for i in range(doy_bins):
            if bin_edges[i] <= doy < bin_edges[i + 1]:
                return self.residual_doy_std.get(str(i), self.residual_std)
        return self.residual_std

    def zscore(self, date_or_doy: str | float | date, observed: float) -> float:
        """
        Compute anomaly z-score:  z = (observed − expected) / σ
        Negative z → NDVI below expected → potential vegetation loss.
        z ≤ -2.0 corresponds to 97.5% confidence of anomaly.
        """
        expected = self.predict(date_or_doy)
        sig      = self.sigma(date_or_doy)
        if sig < 1e-6:
            return 0.0
        return (observed - expected) / sig

    def anomaly_flag(self, date_or_doy: str | float | date, observed: float,
                     thresh: float = _ZSCORE_ALERT_THRESH) -> bool:
        """Return True if the observation is an anomalous NDVI drop."""
        return self.zscore(date_or_doy, observed) <= thresh

    # ────────────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PhenologyModel":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    # ────────────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"PhenologyModel({self.label!r}, var={self.variable}, "
            f"n={self.n_obs}, R²={self.r2:.3f}, σ={self.residual_std:.4f})"
        )

    def expected_curve(self, n_points: int = 365) -> tuple[np.ndarray, np.ndarray]:
        """Return (doy_array, ndvi_array) for the full annual cycle."""
        doys   = np.linspace(1, 365, n_points)
        X      = _design_matrix(doys)
        values = X @ np.array(self.coeffs)
        return doys, values


# ── convenience: fit all records from the division CSV ──────────────────────
def fit_from_csv(csv_path: str | Path, label: str = "division",
                 variable: str = "ndvi") -> PhenologyModel:
    """
    Read `outputs/phenology/ndvi_phenology.csv` and fit a model for the
    given label (default 'division').
    """
    import csv as _csv
    records = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("label", "") == label:
                records.append({
                    "date_label": row["date_label"],
                    "ndvi_mean":  float(row["ndvi_mean"]) if row.get("ndvi_mean") else None,
                    "dw_mean":    float(row["dw_mean"])   if row.get("dw_mean")   else None,
                })
    return PhenologyModel.fit(records, label=label, variable=variable)


# ── CLI: fit + report ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv_arg  = sys.argv[1] if len(sys.argv) > 1 else "outputs/phenology/ndvi_phenology.csv"
    var_arg  = sys.argv[2] if len(sys.argv) > 2 else "ndvi"
    save_arg = sys.argv[3] if len(sys.argv) > 3 else None

    model = fit_from_csv(csv_arg, label="division", variable=var_arg)
    print(f"PhenologyModel({model.label!r}, var={model.variable}, "
          f"n={model.n_obs}, R2={model.r2:.3f}, sigma={model.residual_std:.4f})")
    print(f"  Coeffs: a0={model.coeffs[0]:.4f}  a1={model.coeffs[1]:.4f}  "
          f"b1={model.coeffs[2]:.4f}  a2={model.coeffs[3]:.4f}  b2={model.coeffs[4]:.4f}")

    # Show monthly predictions vs actual
    print("\n  Month  Expected  sigma")
    for month in range(1, 13):
        doy = (date(2024, month, 15).timetuple().tm_yday)
        exp = model.predict(float(doy))
        sig = model.sigma(float(doy))
        print(f"    {month:02d}     {exp:.3f}    +-{sig:.3f}")

    if save_arg:
        model.save(save_arg)
        print(f"\n  Model saved -> {save_arg}")
