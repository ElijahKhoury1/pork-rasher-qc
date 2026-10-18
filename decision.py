"""Tray-level PASS/FAIL decision logic and metrics.

The detector finds individual defects. A production line needs one decision per tray,
so we reduce detections to PASS/FAIL with an explicit, configurable rule:

    FAIL if the tray has at least ``min_defects`` detections of a class in
    ``fail_classes`` with confidence >= ``conf_thresh``; otherwise PASS.

The ground-truth tray label uses the same rule applied to the annotation files, so
the rule itself is a documented assumption, not hidden in the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .data import read_label_class_ids

PASS = "PASS"
FAIL = "FAIL"


@dataclass(frozen=True)
class Detection:
    """One detected defect.

    Attributes:
        cls_name: Class name, e.g. ``"unsealed"``.
        conf: Confidence in [0, 1].
        xyxy: Box corners ``(x1, y1, x2, y2)`` in pixels.
    """

    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


def tray_decision(
    detections: Iterable[Detection],
    conf_thresh: float = 0.25,
    fail_classes: Sequence[str] | None = None,
    min_defects: int = 1,
) -> dict:
    """Decide PASS/FAIL for one tray.

    Args:
        detections: Detections for the tray image.
        conf_thresh: Minimum confidence for a detection to count (0-1).
        fail_classes: Classes that cause a failure. ``None`` means every class.
        min_defects: Number of qualifying detections needed to fail the tray (>= 1).

    Returns:
        Dict with ``status`` ("PASS"/"FAIL"), ``n_defects`` and ``counts`` per class.

    Raises:
        ValueError: If ``conf_thresh`` is outside [0, 1] or ``min_defects`` < 1.
    """
    if not 0.0 <= conf_thresh <= 1.0:
        raise ValueError(f"conf_thresh must be in [0, 1], got {conf_thresh}")
    if min_defects < 1:
        raise ValueError(f"min_defects must be >= 1, got {min_defects}")

    counts: dict[str, int] = {}
    for d in detections:
        if d.conf < conf_thresh:
            continue
        if fail_classes is not None and d.cls_name not in fail_classes:
            continue
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    n = sum(counts.values())
    return {"status": FAIL if n >= min_defects else PASS, "n_defects": n, "counts": counts}


def tray_label_from_yolo_txt(
    label_file: str | Path,
    names: Sequence[str],
    fail_classes: Sequence[str] | None = None,
    min_defects: int = 1,
) -> str:
    """Ground-truth tray label (PASS/FAIL) from a YOLO annotation file.

    Uses the same rule as :func:`tray_decision`, with every annotated box at confidence 1.
    """
    ids = read_label_class_ids(label_file)
    dets = [Detection(names[i], 1.0, (0, 0, 0, 0)) for i in ids]
    return tray_decision(dets, 0.0, fail_classes, min_defects)["status"]


def binary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Precision, recall, F1 and accuracy with FAIL as the positive class.

    Recall on FAIL is the share of defective trays caught; precision on FAIL is the
    share of rejected trays that really were defective (i.e. false-reject cost).

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("No samples to score")
    tp = sum(t == FAIL and p == FAIL for t, p in zip(y_true, y_pred))
    fp = sum(t == PASS and p == FAIL for t, p in zip(y_true, y_pred))
    fn = sum(t == FAIL and p == PASS for t, p in zip(y_true, y_pred))
    tn = sum(t == PASS and p == PASS for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
    }
