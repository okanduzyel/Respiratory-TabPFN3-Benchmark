# Respiratory-TabPFN3-Benchmark

Code and results for the manuscript **"Exposing Patient-Level Data Leakage in Respiratory Sound Classification: A Leakage-Free Benchmark Leveraging Transformer-Based TabPFN-3 and Knowledge Distillation"** (Diagnostics, MDPI; manuscript ID diagnostics-4593104).

The study uses the public ICBHI 2017 Respiratory Sound Database. It compares TabPFN-3 with eight machine learning classifiers under a cycle-level split (leaky) and a patient-level split (leakage-free), and distils TabPFN-3 into small student models.

## Repository structure

| Path | Content | Used for |
|---|---|---|
| `data/respiratory_dataset.csv` | 52 features per respiratory cycle (6,860 cycles, 123 patients, 6 classes) | all experiments |
| `feature_extraction/Dataset_Creator.py` | Extracts the 52 features from the ICBHI audio files with Librosa | creating the CSV above |
| `stage1_cycle_level/` | Cycle-level 5-fold cross-validation (baselines and TabPFN-3) | Stage 1, Tables 2–5, Figure 5 |
| `revision_analysis.py` | Single script for all patient-level analyses | Stages 2 and 3: Tables 1, 6–11, Figure 6, Appendix A (Tables A1–A4, Figure A1) |
| `revision_outputs/` | Saved outputs of `revision_analysis.py` (see below) | reproducing and checking every reported number |
| `legacy/` | Earlier exploratory scripts | **not used** for the reported results |

## Patient-level analysis (`revision_analysis.py`)

The script produces every number of the patient-level part of the revised manuscript from one code base:

- a stratified patient-level split in which every class is present in every training and test fold, repeated three times (15 test folds);
- the eight baselines and TabPFN-3 (Table 6, Table 7, Table A3);
- Nadeau-Bengio corrected resampled t-tests with Holm correction (Tables 8 and 11) and a TOST equivalence test;
- knowledge distillation into an MLP, a small Transformer, XGBoost and decision trees (Table 9) and the hard/soft/combined-label comparison (Table A4);
- feature-group permutation importance (Table 1);
- SHAP values with the SHAP permutation explainer (Figure 6, Figure A1);
- decision rules extracted automatically from the distilled depth-4 tree (Table 10);
- patient-level results with patient-level bootstrap confidence intervals (Table A3).

### How to run

```bash
pip install -r requirements.txt

# local TabPFN (needs the TabPFN model weights)
python revision_analysis.py

# or through the Prior Labs API
export PRIORLABS_TOKEN=<your key>
TABPFN_BACKEND=cloud python revision_analysis.py
```

The results reported in the manuscript were produced with the Prior Labs API (`TABPFN_BACKEND=cloud`, 8 estimators). The script saves its progress after every fold in `revision_outputs/cache/` and continues from there if it is stopped. A quick test run is possible with `QUICK=1`.

Never write an API key into the code; use the `PRIORLABS_TOKEN` environment variable.

### Saved outputs (`revision_outputs/`)

| File | Content |
|---|---|
| `summary.md` | All result tables in readable form |
| `results.json` | All numbers in machine-readable form |
| `fold_assignments.csv` | Patients and cycles per class in every test fold, for all splits |
| `oof_predictions_5fold.csv` | Out-of-fold predictions and class probabilities of TabPFN-3 |
| `cache/*.pkl` | Fold-level predictions and metrics of every model in every fold |
| `fig6_shap_global.png`, `figA_shap_by_class.png` | SHAP figures |
| `student_tree_depth4.txt`, `student_tree_full.txt`, `student_tree_all_leaves.csv` | Distilled decision trees and their leaves |
| `run_log.txt` | Log of the run |

## Stage 1 (cycle-level)

`stage1_cycle_level/1D_All_New_No_Augmentation.py` (eight baselines) and `stage1_cycle_level/1D_tabPFN_STD.py` (TabPFN-3, local `tabpfn` package) read the ICBHI audio files directly. Set `data_dir` in the scripts to the folder of the ICBHI recordings.

## Notes

- `legacy/` contains earlier versions of the experiments. They are kept for transparency only. Their results are **not** reported in the revised manuscript, and some of them use settings that differ from the manuscript.
- The license of TabPFN-3 allows only non-commercial use. The distilled student models are therefore a research proof of concept.
- Data source: Rocha et al., A respiratory sound database for the development of automated classification, ICBHI 2017, https://doi.org/10.1007/978-981-10-7419-6_6.
