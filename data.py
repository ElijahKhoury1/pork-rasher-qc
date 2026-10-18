"""Dataset helpers for the Roboflow YOLO-format pork rasher dataset.

Expected on-disk layout (Roboflow "yolov8" export)::

    <dataset_dir>/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        test/images/*.jpg    test/labels/*.txt

Each label line is ``class_id x_center y_center width height`` (all normalised 0-1).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

WORKSPACE = "hello-2aqe0"
PROJECT = "pork-rasher-error-packaging"
DEFAULT_VERSION = 4


def download_roboflow(
    api_key: str,
    version: int = DEFAULT_VERSION,
    location: str | Path = "datasets/pork-rasher",
    fmt: str = "yolov8",
) -> Path:
    """Download a dataset version from Roboflow Universe.

    Args:
        api_key: Your (free) Roboflow API key. Never commit this to Git.
        version: Dataset version number (the course brief uses Version 4).
        location: Folder to download into.
        fmt: Roboflow export format; ``"yolov8"`` works for Ultralytics.

    Returns:
        Path to the downloaded dataset folder.

    Raises:
        ValueError: If ``api_key`` is empty.
    """
    if not api_key:
        raise ValueError("A Roboflow API key is required.")
    from roboflow import Roboflow  # lazy import: only needed in Colab

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(version).download(fmt, location=str(location))
    return Path(ds.location)


def load_data_yaml(dataset_dir: str | Path) -> dict:
    """Read ``data.yaml`` from a dataset folder.

    Raises:
        FileNotFoundError: If ``data.yaml`` is missing.
    """
    path = Path(dataset_dir) / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")
    with open(path) as f:
        return yaml.safe_load(f)


def class_names(dataset_dir: str | Path) -> list[str]:
    """Return class names ordered by class id."""
    names = load_data_yaml(dataset_dir)["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def list_images(dataset_dir: str | Path, split: str) -> list[Path]:
    """List image files for a split (empty list if the split does not exist)."""
    folder = Path(dataset_dir) / split / "images"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: str | Path) -> Path:
    """Map ``.../images/x.jpg`` to ``.../labels/x.txt``."""
    p = Path(image_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def read_label_class_ids(label_file: str | Path) -> list[int]:
    """Return the class ids of every box in a YOLO label file.

    A missing file or an empty file means "no boxes". Malformed lines raise ValueError
    so that bad annotations are never silently ignored.
    """
    p = Path(label_file)
    if not p.is_file():
        return []
    ids = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{p}:{n}: expected >=5 values, got {len(parts)}")
        ids.append(int(float(parts[0])))
    return ids


def count_classes(dataset_dir: str | Path) -> dict[str, Counter]:
    """Count boxes per class for every split.

    Returns:
        ``{split: Counter({class_name: n_boxes})}``. Useful for spotting class imbalance.
    """
    names = class_names(dataset_dir)
    out: dict[str, Counter] = {}
    for split in SPLITS:
        c: Counter = Counter()
        for img in list_images(dataset_dir, split):
            for cid in read_label_class_ids(label_path_for(img)):
                c[names[cid]] += 1
        out[split] = c
    return out


def base_name(filename: str) -> str:
    """Original frame name of a (possibly augmented) Roboflow file.

    Roboflow appends ``.rf.<hash>`` to every exported/augmented file, so
    ``frame_1_jpg.rf.abc.jpg`` and ``frame_1_jpg.rf.def.jpg`` share the base ``frame_1_jpg``.
    """
    return filename.split(".rf.")[0]


def find_cross_split_leakage(dataset_dir: str | Path) -> dict[str, set[str]]:
    """Find original frames that appear in more than one split.

    Augmented copies of the same frame in train and test would inflate test scores,
    so this should return an empty dict.

    Returns:
        ``{base_name: {splits it appears in}}`` for each leaked frame.
    """
    seen: dict[str, set[str]] = {}
    for split in SPLITS:
        for img in list_images(dataset_dir, split):
            seen.setdefault(base_name(img.name), set()).add(split)
    return {k: v for k, v in seen.items() if len(v) > 1}


def validate_dataset(dataset_dir: str | Path) -> dict:
    """Sanity-check a downloaded dataset and summarise it.

    Checks that every image has a label file, class ids are in range, and reports
    image counts per split.

    Returns:
        Dict with keys ``images`` (per split), ``missing_labels``, ``bad_class_ids``.
    """
    names = class_names(dataset_dir)
    report = {"images": {}, "missing_labels": [], "bad_class_ids": []}
    for split in SPLITS:
        imgs = list_images(dataset_dir, split)
        report["images"][split] = len(imgs)
        for img in imgs:
            lp = label_path_for(img)
            if not lp.is_file():
                report["missing_labels"].append(str(img))
                continue
            for cid in read_label_class_ids(lp):
                if not 0 <= cid < len(names):
                    report["bad_class_ids"].append((str(lp), cid))
    return report
