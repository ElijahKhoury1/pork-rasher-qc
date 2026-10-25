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
| **Input** | RGB image of a packaged tray (`.jpg/.jpeg/.png/.bmp`), resized to 416 px for the selected model (dataset images are 720x540) |
| **Detector output** | List of `Detection(cls_name, conf, xyxy)`: class, confidence in [0,1], box in pixels |
| **Tray output** | `{"status": "PASS"/"FAIL", "n_defects": int, "counts": {class: n}}` |
| **Invalid input** | Missing file -> `FileNotFoundError`; unsupported extension -> `ValueError`; confidence outside [0,1] -> `ValueError` |

**Decision rule (assumption):** a tray FAILS if it has at least `min_defects` (default 1) detections of
a class in `fail_classes` (default: all) with confidence >= `conf_thresh`. The ground-truth tray label is
derived from the annotation files with the same rule. Change `fail_classes` if, for example, `wrinkle`
should not cause a reject.

## Repository layout

```
src/rasher_qc/   data.py, leakage.py, decision.py, train.py, evaluate.py, infer.py
tests/           pytest unit tests (data, decision, leakage) and test_report.html
notebooks/       colab_pipeline.ipynb (full workflow), run_tests.ipynb, generate_docs.ipynb
docs/            HTML API docs generated with pdoc (open docs/index.html)
```

## Local setup

```bash
   pip install -r requirements.txt
   pip install -e .
   pip install pytest pytest-html pdoc
   pytest -v --html=tests/test_report.html --self-contained-html   # writes the test report
   pdoc -o docs src/rasher_qc                                      # generate HTML docs
```

## Workflow

1. `data.download_roboflow` downloads Version 4 in YOLO format.
2. `data.validate_dataset`, `count_classes`, `find_cross_split_leakage` check the data
   (augmented copies of one frame must not sit in both train and test).
3. `leakage.py` removes frames that appear in more than one split (augmented copies of the same photo, matched by file name), so the test images are unseen. It cannot catch near-duplicate frames with different names.
4. `train.train` fine-tunes pretrained YOLO weights. The notebook compares a baseline (`yolov8n`, 640 px)
   against alternatives (`yolov8s`, other image sizes).
5. `evaluate` reports mAP50 / mAP50-95 / per-class AP, tray PASS/FAIL precision/recall/F1,
   and latency, parameters and GFLOPs.
6. The best model is chosen on the **validation** split; the **test** split is used once for the final result.

## Final Model 
Selected model: YOLOv8n at 416 px (n_416), chosen on the validation split (mAP50-95 0.352, 10.5 ms per image on a Tesla T4) over the baseline YOLOv8n at 640 px (0.272) and YOLOv8s at 640 px (0.194). Default settings: confidence threshold 0.25, seed 0, max 25 epochs, early-stopping patience 8, batch size 16. On the test split the tray decision has recall 0.82 and specificity 0.92. The recall target of 0.90 was not met. See the report for full results and limitations.

## Limitations

- One camera and setup, so results may not transfer to other lighting or packaging.
- Rare classes may have few examples; check `count_classes` before trusting per-class AP.
- Tray ground truth is derived from box annotations, not from an independent inspector.

## AI use

Claude was the main AI tool, with Copilot and ChatGPT used occasionally. AI wrote the initial code (the Python package, unit tests, README, licence file and Colab notebooks), explained YOLO, the metrics and the data splits, helped debug Colab and GitHub problems, and gave feedback on report drafts. Approximate contribution: about 100% of the code and 30% of the written report. I checked AI output by reading and running it, asking follow-up questions until I understood it at a fundamental level. The narrowed scope, final epoch count, leakage handling and PASS/FAIL rule were my decisions. AI errors included a bug in its own code, a wrong path to best.pt that broke evaluation, setup errors, a notebook GitHub rejected, and a leakage check that misses near-duplicate frames. Full reflection is in the report.
## Dataset and licence

Code: MIT. Dataset: "Pork Rasher Error(Packaging)" by Hello, Roboflow Universe, CC BY 4.0,
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging>. Not redistributed here.
