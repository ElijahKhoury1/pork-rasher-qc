import pytest

from rasher_qc.decision import (
    FAIL, PASS, Detection, binary_metrics, tray_decision, tray_label_from_yolo_txt,
)


def det(name, conf):
    return Detection(name, conf, (0, 0, 10, 10))


def test_no_detections_passes():
    assert tray_decision([])["status"] == PASS


def test_single_confident_defect_fails():
    r = tray_decision([det("unsealed", 0.9)])
    assert r["status"] == FAIL and r["counts"] == {"unsealed": 1}


def test_low_confidence_ignored():
    assert tray_decision([det("unsealed", 0.1)], conf_thresh=0.25)["status"] == PASS


def test_fail_classes_filter():
    dets = [det("wrinkle", 0.9)]
    assert tray_decision(dets, fail_classes=["unsealed"])["status"] == PASS
    assert tray_decision(dets, fail_classes=["wrinkle"])["status"] == FAIL


def test_min_defects():
    dets = [det("wrinkle", 0.9)]
    assert tray_decision(dets, min_defects=2)["status"] == PASS
    assert tray_decision(dets * 2, min_defects=2)["status"] == FAIL


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_bad_conf_raises(bad):
    with pytest.raises(ValueError):
        tray_decision([], conf_thresh=bad)


def test_bad_min_defects_raises():
    with pytest.raises(ValueError):
        tray_decision([], min_defects=0)


def test_binary_metrics_known_values():
    y_true = [FAIL, FAIL, PASS, PASS]
    y_pred = [FAIL, PASS, FAIL, PASS]
    m = binary_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_binary_metrics_perfect():
    m = binary_metrics([FAIL, PASS], [FAIL, PASS])
    assert m["f1"] == 1.0 and m["accuracy"] == 1.0


def test_binary_metrics_errors():
    with pytest.raises(ValueError):
        binary_metrics([FAIL], [])
    with pytest.raises(ValueError):
        binary_metrics([], [])


def test_label_file_ground_truth(tmp_path):
    names = ["loose-meat", "unsealed"]
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.2 0.2\n")
    assert tray_label_from_yolo_txt(f, names) == FAIL
    assert tray_label_from_yolo_txt(f, names, fail_classes=["loose-meat"]) == PASS
    f.write_text("")
    assert tray_label_from_yolo_txt(f, names) == PASS
    assert tray_label_from_yolo_txt(tmp_path / "missing.txt", names) == PASS
