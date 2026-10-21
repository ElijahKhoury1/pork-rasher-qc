"""Training wrapper around Ultralytics YOLO."""

from __future__ import annotations

from pathlib import Path


def train(
    data_yaml: str | Path,
    weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 0,
    project: str = "runs",
    name: str = "exp",
    **extra,
) -> Path:
    """Fine-tune a pretrained YOLO model on the dataset.

    Args:
        data_yaml: Path to the dataset ``data.yaml``.
        weights: Pretrained checkpoint, e.g. ``yolov8n.pt`` (baseline) or ``yolov8s.pt``.
        epochs: Training epochs.
        imgsz: Training/inference image size (pixels, square).
        batch: Batch size.
        seed: Random seed, so runs are repeatable.
        project: Output folder for runs.
        name: Run name.
        **extra: Any other Ultralytics train argument (e.g. ``patience``, ``hsv_s``).

    Returns:
        Path to ``best.pt`` for the run.
    """
    from ultralytics import YOLO  # lazy import: heavy dependency

    model = YOLO(weights)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        seed=seed, deterministic=True, project=project, name=name, exist_ok=True, **extra,
    )
    return Path(project) / name / "weights" / "best.pt"
