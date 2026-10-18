"""Run a trained model on one image."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .data import IMAGE_EXTS
from .decision import Detection, tray_decision


def predict_detections(model, image_path: str | Path, conf: float = 0.25, imgsz: int = 640) -> list[Detection]:
    """Detect defects in one image.

    Args:
        model: A loaded ``ultralytics.YOLO`` model.
        image_path: Path to an RGB image (.jpg/.jpeg/.png/.bmp).
        conf: Minimum confidence kept by the detector.
        imgsz: Inference size in pixels.

    Returns:
        List of :class:`~rasher_qc.decision.Detection`.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file type is unsupported.
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image type: {p.suffix}")
    res = model.predict(str(p), conf=conf, imgsz=imgsz, verbose=False)[0]
    names = res.names
    return [
        Detection(names[int(c)], float(s), tuple(float(v) for v in xyxy))
        for c, s, xyxy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())
    ]


def inspect_tray(
    model,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    fail_classes: Sequence[str] | None = None,
) -> dict:
    """Full report for one tray image: detections plus PASS/FAIL decision."""
    dets = predict_detections(model, image_path, conf, imgsz)
    report = tray_decision(dets, conf, fail_classes)
    report["detections"] = [d.__dict__ for d in dets]
    return report
