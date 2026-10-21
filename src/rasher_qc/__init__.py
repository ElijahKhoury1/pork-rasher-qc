"""Pork rasher packaging quality inspection (MMA3001 project).

Modules
-------
data
    Dataset download, structure validation, class counts and split-leakage checks.
decision
    Turns detections into a tray-level PASS/FAIL decision and scores that decision.
train
    Thin wrapper around Ultralytics YOLO training.
evaluate
    Detection metrics (mAP), tray-level metrics and latency benchmarking.
infer
    Run a trained model on a single image and return a structured report.
"""

__version__ = "0.1.0"
