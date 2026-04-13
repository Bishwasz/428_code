# 428_code
# Next Best View Selection for Incremental Structure-from-Motion

Undergraduate project comparing a hybrid information-theoretic + coverage-aware
Next Best View (NBV) strategy against a uniform random baseline on a small
Structure-from-Motion reconstruction task.

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`:
  - numpy
  - scipy
  - matplotlib
  - pycolmap

Tested on macOS (Apple Silicon) with Python 3.10.

**Note:** `pycolmap` wraps COLMAP and may require additional system
dependencies (Ceres, Eigen, etc.). See
https://github.com/colmap/pycolmap for platform-specific install notes.

## Setup

```bash
## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset layout

The script expects:
dataset/images/              # 24 input images of the mug
dataset/sparse/0/            # offline full-reconstruction COLMAP model
# (provides GT poses and reference 3D points)

The `sparse/0/` model is used as a pose oracle for candidate scoring and as
the reference point set for the coverage term. It is computed once offline
by running a full COLMAP reconstruction over all 24 images.

## Run

```bash
python nbv.py
```

This runs both strategies (Hybrid-NBV and Random) from a shared 4-image seed
for 8 NBV iterations each, then writes `nbv_vs_random.png` with the
per-iteration comparison.

Total runtime: ~5–10 minutes on a single CPU (dominated by COLMAP matching
and mapping).
