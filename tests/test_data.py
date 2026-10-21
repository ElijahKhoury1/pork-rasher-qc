import pytest
import yaml

from rasher_qc import data


def make_dataset(root, files):
    """files: {(split, filename): label_text or None (no label file)}"""
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["loose-meat", "unsealed"], "nc": 2}))
    for (split, fname), label in files.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / fname).write_bytes(b"x")
        if label is not None:
            stem = fname.rsplit(".", 1)[0]
            (root / split / "labels" / f"{stem}.txt").write_text(label)


def test_base_name_strips_roboflow_suffix():
    assert data.base_name("frame_1_jpg.rf.abc123.jpg") == "frame_1_jpg"
    assert data.base_name("plain.jpg") == "plain.jpg"


def test_class_names_list_and_dict(tmp_path):
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": {1: "b", 0: "a"}}))
    assert data.class_names(tmp_path) == ["a", "b"]


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_data_yaml(tmp_path)


def test_count_classes(tmp_path):
    make_dataset(tmp_path, {
        ("train", "a.jpg"): "0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n",
        ("train", "b.jpg"): "1 .5 .5 .1 .1\n",
        ("test", "c.jpg"): "",
    })
    c = data.count_classes(tmp_path)
    assert c["train"]["loose-meat"] == 1 and c["train"]["unsealed"] == 2
    assert sum(c["test"].values()) == 0


def test_leakage_detected(tmp_path):
    make_dataset(tmp_path, {
        ("train", "frame_1_jpg.rf.aaa.jpg"): "",
        ("test", "frame_1_jpg.rf.bbb.jpg"): "",
        ("test", "frame_2_jpg.rf.ccc.jpg"): "",
    })
    leaks = data.find_cross_split_leakage(tmp_path)
    assert list(leaks) == ["frame_1_jpg"] and leaks["frame_1_jpg"] == {"train", "test"}


def test_no_leakage(tmp_path):
    make_dataset(tmp_path, {("train", "a.jpg"): "", ("test", "b.jpg"): ""})
    assert data.find_cross_split_leakage(tmp_path) == {}


def test_validate_flags_problems(tmp_path):
    make_dataset(tmp_path, {
        ("train", "a.jpg"): None,                # missing label
        ("train", "b.jpg"): "7 .5 .5 .1 .1\n",   # class id out of range
        ("valid", "c.jpg"): "0 .5 .5 .1 .1\n",
    })
    r = data.validate_dataset(tmp_path)
    assert r["images"] == {"train": 2, "valid": 1, "test": 0}
    assert len(r["missing_labels"]) == 1
    assert r["bad_class_ids"][0][1] == 7


def test_malformed_label_raises(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError):
        data.read_label_class_ids(f)


def test_download_requires_key():
    with pytest.raises(ValueError):
        data.download_roboflow("")
