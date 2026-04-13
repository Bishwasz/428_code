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
