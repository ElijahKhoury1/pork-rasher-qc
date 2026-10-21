"""Remove train/valid/test leakage caused by augmented copies of one frame.

Roboflow gives every augmented copy of an original frame a name that starts with
the same stem (``frame_1_jpg.rf.<hash>.jpg``). If copies of one frame land in
different splits, the model is tested on near-identical images it trained on,
which inflates test scores. This module keeps each frame in one split only.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from .data import base_name, find_cross_split_leakage, label_path_for, list_images


def remove_cross_split_leakage(
    dataset_dir: str | Path,
    backup_dir: str | Path = "datasets/removed_leaked",
    priority: Sequence[str] = ("test", "valid", "train"),
) -> int:
    """Move leaked frames out of the lower-priority splits.

    A frame found in several splits is kept in the first split listed in
    ``priority``. Its copies (image and label file) in the other splits are
    moved to ``backup_dir``, not deleted, so the step can be undone.

    Args:
        dataset_dir: Dataset folder containing ``train/``, ``valid/``, ``test/``.
        backup_dir: Where removed files go, grouped by split.
        priority: Splits in the order they keep a leaked frame.

    Returns:
        Number of images moved.
    """
    backup = Path(backup_dir)
    moved = 0
    for base, splits in find_cross_split_leakage(dataset_dir).items():
        keep = next(s for s in priority if s in splits)
        for split in splits - {keep}:
            for img in list_images(dataset_dir, split):
                if base_name(img.name) != base:
                    continue
                for f in (img, label_path_for(img)):
                    if f.exists():
                        dest = backup / split / f.parent.name
                        dest.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(f), dest / f.name)
                moved += 1
    return moved
