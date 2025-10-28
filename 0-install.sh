#!/bin/bash
# Create an environment (Python 3.10 is a safe bet)
conda create -n bayesflow python=3.10 -y
conda activate bayesflow

# Upgrade pip
python -m pip install --upgrade pip

# Install BayesFlow and dependencies
# Linux/Windows (CPU):
pip install bayesflow tensorflow

# macOS Apple Silicon (M1/M2/M3) — better performance:
# pip install bayesflow tensorflow-macos tensorflow-metal

# (Optional) quick check
python - <<'PY'
import bayesflow, sys
try:
    import tensorflow as tf
    print("BayesFlow:", bayesflow.__version__)
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("Import test error:", e, file=sys.stderr)
PY
