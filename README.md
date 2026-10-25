# Pork Rasher Packaging Quality Inspection

MMA3001 individual project. Detects and localises packaging defects in RGB images of pork rasher
trays (`loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`) with a YOLO object
detector, then turns the detections into a **tray-level PASS/FAIL** decision.

## Quick start (Google Colab)

1. Push this repo to your GitHub account (see below).
2. Open `notebooks/colab_pipeline.ipynb` in Colab: **File > Open notebook > GitHub**, paste your repo URL.
3. Set **Runtime > Change runtime type > GPU (T4)**.
4. Edit `REPO_URL` in the first code cell.
5. Add your Roboflow API key as a Colab secret named `ROBOFLOW_API_KEY` (key icon in the left sidebar).
   Get a free key at <https://app.roboflow.com> (Settings > API Keys). **Never commit the key.**
6. Run all cells.

## Inputs and outputs

| | Description |
|---|---|
| **Input** | RGB image of a packaged tray (`.jpg/.jpeg/.png/.bmp`), resized to 640 px (dataset images are 720x540) |
| **Detector output** | List of `Detection(cls_name, conf, xyxy)`: class, confidence in [0,1], box in pixels |
| **Tray output** | `{"status": "PASS"/"FAIL", "n_defects": int, "counts": {class: n}}` |
| **Invalid input** | Missing file -> `FileNotFoundError`; unsupported extension -> `ValueError`; confidence outside [0,1] -> `ValueError` |

**Decision rule (assumption):** a tray FAILS if it has at least `min_defects` (default 1) detections of
a class in `fail_classes` (default: all) with confidence >= `conf_thresh`. The ground-truth tray label is
derived from the annotation files with the same rule. Change `fail_classes` if, for example, `wrinkle`
should not cause a reject.

## Repository layout

```
src/rasher_qc/   data.py, decision.py, train.py, evaluate.py, infer.py
tests/           pytest unit tests (no GPU or dataset needed)
notebooks/       colab_pipeline.ipynb  (full workflow)
docs/            HTML API docs (generated, see below)
```

## Local setup

```bash
pip install -r requirements.txt
pip install -e .
pytest -v                                   # run tests
pdoc -o docs src/rasher_qc                  # generate HTML docs
pytest --html=test_report.html --self-contained-html   # optional, needs pytest-html
```

## Workflow

1. `data.download_roboflow` downloads Version 4 in YOLO format.
2. `data.validate_dataset`, `count_classes`, `find_cross_split_leakage` check the data
   (augmented copies of one frame must not sit in both train and test).
3. `train.train` fine-tunes pretrained YOLO weights. The notebook compares a baseline (`yolov8n`, 640 px)
   against alternatives (`yolov8s`, other image sizes).
4. `evaluate` reports mAP50 / mAP50-95 / per-class AP, tray PASS/FAIL precision/recall/F1,
   and latency, parameters and GFLOPs.
5. The best model is chosen on the **validation** split; the **test** split is used once for the final result.

## Limitations

- One camera and setup, so results may not transfer to other lighting or packaging.
- Rare classes may have few examples; check `count_classes` before trusting per-class AP.
- Tray ground truth is derived from box annotations, not from an independent inspector.

## AI use

AI was used to assist with code generation and writing the report. Also used to make amends and fix code to ensure that everything was functional

## Dataset and licence

Code: MIT. Dataset: "Pork Rasher Error(Packaging)" by Hello, Roboflow Universe, CC BY 4.0,
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging>. Not redistributed here.
