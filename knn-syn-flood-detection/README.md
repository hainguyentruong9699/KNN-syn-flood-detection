# KNN SYN Flood Detection

A student project implementing **K-Nearest Neighbors (KNN) from scratch** to classify network-flow records as **SYN Flood** or **Benign** traffic. The implementation uses Python, NumPy, and pandas; it does not use scikit-learn's KNN classifier.

> **Project scope:** This is an **offline, dataset-based proof of concept**, not a real-time intrusion detection or DDoS mitigation product. Detection results must be validated before making performance claims.

## Repository structure

```text
knn-syn-flood-detection/
├── src/
│   └── knn_from_scratch.py
├── data/
│   ├── syn_train.csv
│   └── syn_test.csv
├── docs/
│   └── presentation.pptx
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Features

- KNN implemented directly using NumPy (Euclidean and Manhattan distance).
- Uniform or distance-weighted voting.
- Z-score feature scaling based on training-set statistics.
- K selection with 5-fold cross-validation using mean F1-score.
- Evaluation: confusion matrix counts (TP, TN, FP, FN), accuracy, precision, recall, F1-score, false alarm rate, and false negative rate.
- CSV exports of per-record results and misclassified flows.

## Requirements

- Python 3.10+ (recommended)
- Dependencies listed in `requirements.txt`

## Setup and run

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The original script uses relative filenames `syn_train.csv` and `syn_test.csv` in the **current working directory**. Run it from `data/` so those paths resolve:

```bash
cd data
python ../src/knn_from_scratch.py
```

The script writes `ket_qua_du_doan.csv` and `mau_du_doan_sai.csv` to the working directory. These generated files are ignored by Git.

## Data

- `data/syn_train.csv`: 20,000 network-flow rows.
- `data/syn_test.csv`: 908 rows after the header, including **one entirely blank row**.
- The `Label` field uses `Syn` and `Benign`.
- The slides discuss the CICDDoS2019 dataset. **Verify the exact provenance, filtering, transformations, redistribution terms, and permission to publish the provided CSVs** before making a dataset-origin or license claim.

## Known limitations / work to do

1. **Blank test row:** `syn_test.csv` has one empty row. The current script does not validate missing values; preprocessing is needed before relying on its predictions.
2. **Train/test overlap:** Some rows overlap between the train and test sets, potentially inflating evaluation results. Rebuild a verified independent split.
3. **Cross-validation leakage:** The script scales the full training set *before* creating CV folds. Fit scaling parameters on each fold's training portion instead.
4. **Label validation:** Any value other than `Syn` is currently assigned label `0`; unknown values should be rejected, not silently treated as benign.
5. **Resource usage:** Pairwise distance matrices used for CV can consume significant memory on larger data.
6. **Reproducibility:** Record precise dependency versions, experiment details, and verified test metrics before reporting detection performance.

**No benchmark accuracy is claimed in this README.** The code is included as provided and has not been refactored or functionally corrected for the issues above.

## Documentation and authorship

The presentation in `docs/` is coursework material. Clarify personal contributions and obtain permission from the other contributors before publicly sharing any group-owned content. Confirm that the data can be redistributed and that the MIT software license is appropriate for the code you have rights to license.

## License

The repository includes an MIT license template for software. Review the copyright holder(s) and permissions before publication; this does not override third-party dataset or presentation rights.
