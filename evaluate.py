"""Evaluation: detection mAP, tray-level metrics, and speed/size benchmarking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import class_names, label_path_for, list_images
from .decision import binary_metrics, tray_decision, tray_label_from_yolo_txt
from .infer import predict_detections


def detection_metrics(weights: str | Path, data_yaml: str | Path, split: str = "test", imgsz: int = 640) -> dict:
    """Object-detection metrics on a labelled split.

    Returns:
        Dict with ``map50``, ``map50_95``, ``precision``, ``recall`` and ``per_class_map50_95``.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, verbose=False)
    names = m.names
    per_class = {names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
    return {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class_map50_95": per_class,
    }


def tray_level_metrics(
    weights: str | Path,
    dataset_dir: str | Path,
    split: str = "test",
    conf: float = 0.25,
    imgsz: int = 640,
    fail_classes: Sequence[str] | None = None,
) -> dict:
    """PASS/FAIL precision/recall/F1 of the full pipeline against annotation-derived labels."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = class_names(dataset_dir)
    y_true, y_pred = [], []
    for img in list_images(dataset_dir, split):
        y_true.append(tray_label_from_yolo_txt(label_path_for(img), names, fail_classes))
        dets = predict_detections(model, img, conf, imgsz)
        y_pred.append(tray_decision(dets, conf, fail_classes)["status"])
    return binary_metrics(y_true, y_pred)


def benchmark_latency(
    weights: str | Path,
    images: Sequence[str | Path],
    imgsz: int = 640,
    warmup: int = 5,
    runs: int = 50,
) -> dict:
    """Measure inference latency and model size.

    Args:
        weights: Path to a trained ``.pt`` file.
        images: Images to cycle through (need at least one).
        imgsz: Inference size.
        warmup: Untimed runs, so GPU/cache start-up does not bias results.
        runs: Timed runs.

    Returns:
        Dict with ``mean_ms``, ``p95_ms``, ``fps``, ``params_m`` and ``gflops``.
    """
    if not images:
        raise ValueError("Need at least one image to benchmark")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = [str(i) for i in images]
    for i in range(warmup):
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, verbose=False)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    info = model.info(verbose=False)  # (layers, params, gradients, GFLOPs)
    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "fps": float(1000 / np.mean(times)),
        "params_m": float(info[1]) / 1e6,
        "gflops": float(info[3]),
    }
