import yaml

from rasher_qc import data
from rasher_qc.leakage import remove_cross_split_leakage


def make_dataset(root, files):
    """files: {(split, filename): label_text}"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["loose-meat", "unsealed"]}))
    for (split, fname), label in files.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / fname).write_bytes(b"x")
        stem = fname.rsplit(".", 1)[0]
        (root / split / "labels" / f"{stem}.txt").write_text(label)


def test_leaked_copy_removed_from_lower_priority_split(tmp_path):
    ds = tmp_path / "ds"
    make_dataset(ds, {
        ("train", "frame_1_jpg.rf.aaa.jpg"): "0 .5 .5 .1 .1\n",
        ("test", "frame_1_jpg.rf.bbb.jpg"): "0 .5 .5 .1 .1\n",
        ("train", "frame_2_jpg.rf.ccc.jpg"): "",
    })
    moved = remove_cross_split_leakage(ds, backup_dir=tmp_path / "backup")
    assert moved == 1
    assert data.find_cross_split_leakage(ds) == {}
    assert len(data.list_images(ds, "test")) == 1    # test copy kept
    assert len(data.list_images(ds, "train")) == 1   # only frame_2 left
    assert (tmp_path / "backup" / "train" / "images" / "frame_1_jpg.rf.aaa.jpg").exists()
    assert (tmp_path / "backup" / "train" / "labels" / "frame_1_jpg.rf.aaa.txt").exists()


def test_no_leakage_moves_nothing(tmp_path):
    ds = tmp_path / "ds"
    make_dataset(ds, {("train", "a.jpg"): "", ("test", "b.jpg"): ""})
    assert remove_cross_split_leakage(ds, backup_dir=tmp_path / "backup") == 0
    assert len(data.list_images(ds, "train")) == 1


def test_valid_beats_train(tmp_path):
    ds = tmp_path / "ds"
    make_dataset(ds, {
        ("train", "f_jpg.rf.a.jpg"): "",
        ("valid", "f_jpg.rf.b.jpg"): "",
    })
    remove_cross_split_leakage(ds, backup_dir=tmp_path / "backup")
    assert len(data.list_images(ds, "valid")) == 1
    assert len(data.list_images(ds, "train")) == 0
