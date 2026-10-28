# -*- coding: utf-8 -*-
"""
revision_analysis.py
====================
Single, reproducible pipeline for the major revision (Diagnostics).
It produces every number that the revised manuscript needs, from one code base.

Run (from the folder that contains respiratory_dataset.csv):
    python revision_analysis.py

Environment variables (all optional):
    TABPFN_BACKEND   local (default) | cloud | mock (mock = fast stand-in, only for testing the script)
    TABPFN_DEVICE    cpu (default) | mps | cuda          (local backend only)
    PRIORLABS_TOKEN  API key                              (cloud backend only; never hard-code it)
    SHAP_N           explained test cycles per fold (default 30)
    SHAP_BG          background cycles per fold (default 20)
    QUICK=1          very small run to check that everything works

The script saves its progress after every fold (revision_outputs/cache). If it stops,
run it again and it continues where it stopped.

Outputs (revision_outputs/):
    summary.md, results.json, fold_assignments.csv, oof_predictions_5fold.csv,
    fig6_shap_global.png, figA_shap_by_class.png, student_tree_full.txt, run_log.txt
"""
import os, sys, json, time, pickle, math, warnings, platform
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (classification_report, confusion_matrix, matthews_corrcoef,
                             accuracy_score, balanced_accuracy_score, f1_score)
from xgboost import XGBClassifier, XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
INPUT_CSV = os.environ.get("INPUT_CSV", os.path.join("data", "respiratory_dataset.csv"))
TARGET_COL = "label_id"
DROP_COLS = ["filename", "patient_id", "label", "split"]
OUT = "revision_outputs"
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

BACKEND = os.environ.get("TABPFN_BACKEND", "local").lower()
DEVICE = os.environ.get("TABPFN_DEVICE", "cpu")
QUICK = os.environ.get("QUICK", "0") == "1"
SHAP_N = int(os.environ.get("SHAP_N", "6" if QUICK else "30"))
SHAP_BG = int(os.environ.get("SHAP_BG", "5" if QUICK else "20"))
N_FOLDS, N_REPEATS = 5, 3
N_BOOT = 200 if QUICK else 2000
EQ_MARGIN = 0.05          # equivalence margin for TOST on MCC (also the 90% CI is reported)
STUDENT_EPOCHS = 3 if QUICK else 40

LABELS = {'bronchiectasis': 0, 'bronchiolitis': 1, 'copd': 2, 'healthy': 3, 'pneumonia': 4, 'urti': 5}
CLASS_NAMES = list(LABELS.keys())
NICE = {'bronchiectasis': 'Bronchiectasis', 'bronchiolitis': 'Bronchiolitis', 'copd': 'COPD',
        'healthy': 'Healthy', 'pneumonia': 'Pneumonia', 'urti': 'URTI'}
K = len(CLASS_NAMES)

LOG = open(os.path.join(OUT, "run_log.txt"), "a")
T0 = time.time()


def log(*a):
    s = f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:6.1f} min] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
X_all = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore").astype(np.float32)
FEATS = list(X_all.columns)
y_all = df[TARGET_COL].to_numpy().astype(int)
groups = df["patient_id"].to_numpy()
assert X_all.shape[1] == 52, X_all.shape
if QUICK:
    _pc = pd.Series(y_all).groupby(groups).first()
    _sel = _pc.groupby(_pc).head(6).index
    keep = np.isin(groups, _sel)
    X_all, y_all, groups = X_all[keep].reset_index(drop=True), y_all[keep], groups[keep]
log(f"Data: {len(y_all)} cycles, {len(np.unique(groups))} patients, {X_all.shape[1]} features, backend={BACKEND}")

# ----------------------------------------------------------------------------
# TabPFN backend
# ----------------------------------------------------------------------------
TABPFN_INFO = {"backend": BACKEND}


def make_tabpfn():
    if BACKEND == "local":
        from tabpfn import TabPFNClassifier
        try:
            import tabpfn; TABPFN_INFO["version"] = getattr(tabpfn, "__version__", "?")
        except Exception:
            pass
        try:
            return TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True, random_state=0)
        except TypeError:
            return TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True)
    if BACKEND == "cloud":
        import tabpfn_client
        from tabpfn_client import TabPFNClassifier
        tabpfn_client.set_access_token(os.environ["PRIORLABS_TOKEN"])
        TABPFN_INFO["version"] = "tabpfn_client " + getattr(tabpfn_client, "__version__", "?")
        return TabPFNClassifier(n_estimators=8)
    if BACKEND == "mock":
        from sklearn.ensemble import HistGradientBoostingClassifier
        TABPFN_INFO["version"] = "MOCK (HistGradientBoosting) - test only"
        return HistGradientBoostingClassifier(max_iter=60, random_state=0)
    raise ValueError(BACKEND)


def with_retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network problems with the cloud backend
            if i == tries - 1:
                raise
            w = 15 * 2 ** i
            log(f"  ! error: {e!r}; retry in {w}s")
            time.sleep(w)


def full_proba(model, X):
    """Probabilities for all K classes, in label order 0..K-1."""
    p = with_retry(lambda: model.predict_proba(X))
    cls = list(getattr(model, "classes_", range(p.shape[1])))
    out = np.zeros((len(X), K), dtype=np.float64)
    for j, c in enumerate(cls):
        out[:, int(c)] = p[:, j]
    return out


# ----------------------------------------------------------------------------
# Baseline classifiers (settings as reported in the manuscript, Section 2.3)
# ----------------------------------------------------------------------------
def make_baselines():
    return {
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "Logistic Regression": LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=100, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma=0.01, class_weight="balanced",
                                                   random_state=42)),
        "KNN": KNeighborsClassifier(metric="manhattan", n_neighbors=3, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=30, min_samples_split=2,
                                                min_samples_leaf=1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3,
                             random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=2,
                                                min_samples_leaf=1, criterion="entropy", bootstrap=False,
                                                random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(colsample_bytree=0.8, gamma=0, learning_rate=0.2, max_depth=4, n_estimators=200,
                                 subsample=0.8, objective="multi:softprob", random_state=42,
                                 eval_metric="mlogloss", n_jobs=-1),
    }


BASELINES = list(make_baselines().keys())

# ----------------------------------------------------------------------------
# Student models (as in the distillation code used for Table 9)
# ----------------------------------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, k))

    def forward(self, x):
        return self.net(x)


class SmallTransformer(nn.Module):
    def __init__(self, d, k, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(d, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, k)

    def forward(self, x):
        return self.fc(self.transformer(self.embedding(x.unsqueeze(1))).squeeze(1))


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def train_student(model, X, y, P, epochs, lr, alpha, seed):
    """loss = alpha * KL(teacher || student) + (1 - alpha) * CE(y, student); alpha=0 -> hard labels only."""
    torch.manual_seed(seed); np.random.seed(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce, kl = nn.CrossEntropyLoss(), nn.KLDivLoss(reduction="batchmean")
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long),
                       torch.tensor(P, dtype=torch.float32))
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=32, shuffle=True, generator=g)
    model.train()
    for _ in range(epochs):
        for xb, yb, pb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = alpha * kl(torch.log_softmax(logits, 1), pb) + (1 - alpha) * ce(logits, yb)
            loss.backward(); opt.step()
    model.eval()
    return model


def predict_student(model, X):
    with torch.no_grad():
        return torch.argmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()


def run_students(Xtr, ytr, Ptr, Xte, seed, ablation=False):
    out = {}
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
                                            n_jobs=-1))
    xgb.fit(Xtr, Ptr); out["Student XGBoost"] = np.argmax(xgb.predict(Xte), 1)
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(Xtr, Ptr); out["Student Decision Tree"] = np.argmax(dt.predict(Xte), 1)
    dt4 = DecisionTreeRegressor(max_depth=4, random_state=42)   # short, readable rules (Table 10)
    dt4.fit(Xtr, Ptr); out["Student Decision Tree (depth 4)"] = np.argmax(dt4.predict(Xte), 1)
    out["Student MLP"] = predict_student(
        train_student(SmallMLP(Xtr.shape[1], K), Xtr, ytr, Ptr, STUDENT_EPOCHS, 0.005, 0.5, seed), Xte)
    out["Student Transformer"] = predict_student(
        train_student(SmallTransformer(Xtr.shape[1], K), Xtr, ytr, Ptr, STUDENT_EPOCHS, 0.002, 0.5, seed + 1), Xte)
    if ablation:
        out["MLP hard labels only (alpha=0)"] = predict_student(
            train_student(SmallMLP(Xtr.shape[1], K), Xtr, ytr, Ptr, STUDENT_EPOCHS, 0.005, 0.0, seed), Xte)
        out["MLP soft labels only (alpha=1)"] = predict_student(
            train_student(SmallMLP(Xtr.shape[1], K), Xtr, ytr, Ptr, STUDENT_EPOCHS, 0.005, 1.0, seed), Xte)
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def fold_metrics(y, p):
    rep = classification_report(y, p, labels=list(range(K)), target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(K)))
    spec = {}
    for c in range(K):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        spec[CLASS_NAMES[c]] = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y, p), "mcc": matthews_corrcoef(y, p),
        "macro_precision": rep["macro avg"]["precision"], "macro_recall": rep["macro avg"]["recall"],
        "macro_f1": rep["macro avg"]["f1-score"], "weighted_f1": rep["weighted avg"]["f1-score"],
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"],
                          "specificity": spec[c], "support": int(rep[c]["support"])} for c in CLASS_NAMES},
    }


def ms(v, d=4):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f}±{np.nanstd(v):.{d}f}"


def nb_test(a, b, ratio):
    """Nadeau-Bengio corrected resampled t-test, J folds, rho = n_test/n_train."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    se = math.sqrt((1.0 / J + ratio) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), J - 1)
    return d.mean(), t, p, se


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, p[i] * (m - r))); adj[i] = run
    return adj


def tost(a, b, ratio, delta):
    d = np.asarray(a) - np.asarray(b); J = len(d)
    mean, _, _, se = nb_test(a, b, ratio)
    tc = stats.t.ppf(0.95, J - 1)
    p_low = stats.t.sf((mean + delta) / se, J - 1)
    p_up = stats.t.cdf((mean - delta) / se, J - 1)
    return {"mean_diff": mean, "ci90_low": mean - tc * se, "ci90_high": mean + tc * se, "margin": delta,
            "p_tost": max(p_low, p_up), "equivalent": bool(max(p_low, p_up) < 0.05)}


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
def split_sgkf(seed=42):
    """Patient-level Stratified Group 5-fold, as in the manuscript (Tables 6, 7, 9)."""
    return list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                .split(X_all, y_all, groups=groups))


def split_repeated(seed):
    """Patient-level stratified 5-fold that really changes between repetitions.
    Patients of each class are shuffled with the given seed and dealt to the folds in turn.
    (StratifiedGroupKFold(shuffle=True) gives almost the same folds for different seeds here,
    because every patient belongs to one class.)"""
    rng = np.random.RandomState(seed)
    pat_class = pd.Series(y_all).groupby(groups).first()
    fold_of = {}
    for c in range(K):
        pts = pat_class[pat_class == c].index.to_numpy().copy()
        rng.shuffle(pts)
        start = rng.randint(N_FOLDS)
        for i, pt in enumerate(pts):
            fold_of[pt] = (start + i) % N_FOLDS
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(N_FOLDS)]


# ----------------------------------------------------------------------------
# Feature-group permutation importance (Table 1)
# ----------------------------------------------------------------------------
GROUPS = {
    "Spectral Domain Metrics": [c for c in FEATS if c.startswith("spectral_")],
    "Adventitious Sounds": [c for c in FEATS if c in ("crackle", "wheeze")],
    "Spectral Cepstral (Mean)": [c for c in FEATS if c.startswith("mfcc_mean")],
    "Temporal Domain Metrics": [c for c in FEATS if c.startswith("zcr") or c.startswith("rms")],
    "Spectral Cepstral (Std)": [c for c in FEATS if c.startswith("mfcc_std")],
}


def group_permutation(model, Xte, yte, seed, n_rep=3):
    base_p = np.argmax(full_proba(model, Xte), 1)
    base_acc = accuracy_score(yte, base_p); base_mcc = matthews_corrcoef(yte, base_p)
    rng = np.random.RandomState(seed); drops_acc, drops_mcc = {}, {}
    for g, cols in GROUPS.items():
        da, dm = [], []
        for _ in range(n_rep):
            Xp = Xte.copy()
            for c in cols:
                Xp[c] = rng.permutation(Xp[c].values)
            pp = np.argmax(full_proba(model, Xp), 1)
            da.append(base_acc - accuracy_score(yte, pp)); dm.append(base_mcc - matthews_corrcoef(yte, pp))
        drops_acc[g] = max(0.0, float(np.mean(da))); drops_mcc[g] = max(0.0, float(np.mean(dm)))
    tot = sum(drops_acc.values()) or 1.0
    return {g: drops_acc[g] / tot for g in GROUPS}, drops_acc, drops_mcc


# ----------------------------------------------------------------------------
# SHAP for TabPFN (5-fold scheme)
# ----------------------------------------------------------------------------
def shap_fold(model, Xtr, Xte, yte, seed):
    import shap
    rng = np.random.RandomState(seed)
    bg = Xtr.iloc[rng.choice(len(Xtr), min(SHAP_BG, len(Xtr)), replace=False)]
    per = max(1, int(math.ceil(SHAP_N / K)))
    idx = []
    for c in range(K):
        ic = np.where(yte == c)[0]
        if len(ic):
            idx += list(rng.choice(ic, min(per, len(ic)), replace=False))
    rest = np.setdiff1d(np.arange(len(yte)), idx)
    if len(idx) < SHAP_N and len(rest):
        idx += list(rng.choice(rest, min(SHAP_N - len(idx), len(rest)), replace=False))
    idx = np.array(idx[:max(SHAP_N, len(idx))])
    cols = list(Xtr.columns)

    def f(a):
        return full_proba(model, pd.DataFrame(np.asarray(a, dtype=np.float32), columns=cols))

    masker = shap.maskers.Independent(bg.values, max_samples=len(bg))
    expl = shap.PermutationExplainer(f, masker, seed=seed)
    sv = expl(Xte.iloc[idx].values, max_evals=2 * (2 * len(cols)) + 1, silent=True)
    vals = np.asarray(sv.values)              # (n, features, classes)
    return vals, yte[idx]


# ----------------------------------------------------------------------------
# Per-fold work (cached)
# ----------------------------------------------------------------------------
def fold_cache(name):
    return os.path.join(CACHE, name + ".pkl")


def run_fold(scheme, fkey, tr, te, seed, do_shap=False, do_perm=False, ablation=False):
    path = fold_cache(f"{scheme}_{fkey}")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    Xtr, Xte = X_all.iloc[tr], X_all.iloc[te]
    ytr, yte = y_all[tr], y_all[te]
    res = {"test_idx": te, "n_train": len(tr), "n_test": len(te), "pred": {}, "metrics": {}}
    t = time.time()
    teacher = make_tabpfn()
    with_retry(lambda: teacher.fit(Xtr, ytr))
    P_te = full_proba(teacher, Xte)
    P_tr = full_proba(teacher, Xtr)          # soft labels for the students (training patients only)
    res["proba_tabpfn"] = P_te
    res["pred"]["TabPFN-3"] = np.argmax(P_te, 1)
    log(f"  {scheme} {fkey}: TabPFN done ({time.time()-t:.0f}s)")
    if do_perm:
        res["perm_norm"], res["perm_acc"], res["perm_mcc"] = group_permutation(teacher, Xte, yte, seed)
    if do_shap:
        t = time.time()
        res["shap_vals"], res["shap_y"] = shap_fold(teacher, Xtr, Xte, yte, seed)
        log(f"  {scheme} {fkey}: SHAP done ({time.time()-t:.0f}s)")
    for name, clf in make_baselines().items():
        clf.fit(Xtr.values, ytr)
        res["pred"][name] = clf.predict(Xte.values)
    res["pred"].update(run_students(Xtr.values, ytr, P_tr, Xte.values, seed, ablation=ablation))
    for m, p in res["pred"].items():
        res["metrics"][m] = fold_metrics(yte, p)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    log(f"  {scheme} {fkey}: saved (MCC TabPFN={res['metrics']['TabPFN-3']['mcc']:.4f})")
    return res


# ============================================================================
# 1) 5-fold patient-level scheme (descriptive tables, SHAP)
# ============================================================================
log("=== Part 1: patient-level Stratified Group 5-fold (seed 42) ===")
S5 = split_sgkf(42)
F5 = [run_fold("sgkf5", f"f{k+1}", tr, te, 100 + k, do_shap=True) for k, (tr, te) in enumerate(S5)]

# ============================================================================
# 2) repeated 3x5 scheme (statistical tests, permutation importance, ablation)
# ============================================================================
log("=== Part 2: repeated 3x5 patient-level CV ===")
REP = []
for r in range(N_REPEATS):
    for k, (tr, te) in enumerate(split_repeated(2026 + r)):
        REP.append(run_fold("rep3x5", f"r{r+1}f{k+1}", tr, te, 1000 + 10 * r + k, do_perm=True, ablation=True))
RATIO = float(np.mean([f["n_test"] / f["n_train"] for f in REP]))

# ============================================================================
# 3) pooled decision rules from the distilled tree (Table 10)
# ============================================================================
log("=== Part 3: decision rules from the distilled tree (pooled data) ===")
rule_path = fold_cache("pooled_teacher")
if os.path.exists(rule_path):
    P_all = pickle.load(open(rule_path, "rb"))
else:
    tch = make_tabpfn(); with_retry(lambda: tch.fit(X_all, y_all)); P_all = full_proba(tch, X_all)
    pickle.dump(P_all, open(rule_path, "wb"))
tree10 = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_all.values, P_all)
with open(os.path.join(OUT, "student_tree_full.txt"), "w") as fh:
    fh.write(export_text(tree10, feature_names=FEATS, decimals=4, max_depth=10))
tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X_all.values, P_all)
with open(os.path.join(OUT, "student_tree_depth4.txt"), "w") as fh:
    fh.write(export_text(tree, feature_names=FEATS, decimals=4))
R_TREE_FIDELITY = float(np.mean(np.argmax(tree.predict(X_all.values), 1) == np.argmax(P_all, 1)))
leaf_of = tree.apply(X_all.values)
tr_ = tree.tree_
parent = {}
for node in range(tr_.node_count):
    for child, side in ((tr_.children_left[node], "le"), (tr_.children_right[node], "gt")):
        if child != -1:
            parent[child] = (node, side)


def leaf_rule(leaf):
    bounds = {}
    n = leaf
    while n in parent:
        p, side = parent[n]
        f, thr = FEATS[tr_.feature[p]], tr_.threshold[p]
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        bounds[f] = (lo, min(hi, thr)) if side == "le" else (max(lo, thr), hi)
        n = p
    parts = []
    for f, (lo, hi) in bounds.items():
        if f in ("crackle", "wheeze"):
            parts.append(f"{f.capitalize()} = {0 if hi < 0.5 else 1}")
        elif np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"{lo:.4g} < {f} ≤ {hi:.4g}")
        elif np.isfinite(hi):
            parts.append(f"{f} ≤ {hi:.4g}")
        else:
            parts.append(f"{f} > {lo:.4g}")
    return " AND ".join(parts), len(parts)


rules = []
priors = {c: float(np.mean(y_all == c)) for c in range(K)}
for leaf in np.unique(leaf_of):
    mask = leaf_of == leaf
    cls = int(np.argmax(np.asarray(tr_.value[leaf]).ravel()))
    sup = int(mask.sum()); conf = float(np.mean(y_all[mask] == cls))
    txt, nc = leaf_rule(leaf)
    rules.append({"leaf": int(leaf), "class": NICE[CLASS_NAMES[cls]], "support": sup, "confidence": conf,
                  "lift": conf / priors[cls] if priors[cls] else np.nan, "n_conditions": nc, "rule": txt})
rules_df = pd.DataFrame(rules)
rules_df.to_csv(os.path.join(OUT, "student_tree_all_leaves.csv"), index=False)
MIN_SUP = 5 if QUICK else 20
rep_rules = []
for c in CLASS_NAMES:
    sub = rules_df[(rules_df["class"] == NICE[c]) & (rules_df.support >= MIN_SUP)]
    if len(sub):
        rep_rules.append(sub.sort_values(["confidence", "support"], ascending=False).iloc[0].to_dict())

# ============================================================================
# 4) Summaries
# ============================================================================
log("=== Part 4: summaries ===")
R = {"info": {"backend": TABPFN_INFO, "python": platform.python_version(), "torch": torch.__version__,
              "n_cycles": int(len(y_all)), "n_patients": int(len(np.unique(groups))), "quick": QUICK,
              "shap_n_per_fold": SHAP_N, "shap_bg": SHAP_BG, "ratio_test_train_rep": RATIO,
              "student_params": {"Student MLP": n_params(SmallMLP(52, K)),
                                 "Student Transformer": n_params(SmallTransformer(52, K))}}}
L = []   # markdown lines
L.append(f"# Revision analysis results\n\nBackend: {TABPFN_INFO}  \nCycles: {len(y_all)}, patients: "
         f"{len(np.unique(groups))}\n")

# --- fold distribution (Table A2) ---
rows = []
for name, splits in [("sgkf5", S5)] + [(f"rep{r+1}", split_repeated(2026 + r)) for r in range(N_REPEATS)]:
    for k, (tr, te) in enumerate(splits):
        for c in range(K):
            m = y_all[te] == c
            rows.append({"scheme": name, "fold": k + 1, "class": NICE[CLASS_NAMES[c]],
                         "test_patients": int(len(np.unique(groups[te][m]))), "test_cycles": int(m.sum()),
                         "train_patients": int(len(np.unique(groups[tr][y_all[tr] == c])))})
fd = pd.DataFrame(rows); fd.to_csv(os.path.join(OUT, "fold_assignments.csv"), index=False)
R["all_classes_in_every_fold"] = bool((fd.test_patients > 0).all() and (fd.train_patients > 0).all())
t5 = fd[fd.scheme == "sgkf5"]
L.append("## Table A2: patients / cycles per class in each test fold (Stratified Group 5-fold, seed 42)\n")
L.append("| Class | " + " | ".join(f"Fold {k}" for k in range(1, 6)) + " |\n|---|" + "---|" * 5)
for c in CLASS_NAMES:
    s = t5[t5["class"] == NICE[c]].sort_values("fold")
    L.append(f"| {NICE[c]} | " + " | ".join(f"{a} / {b}" for a, b in zip(s.test_patients, s.test_cycles)) + " |")
L.append(f"\nEvery class present in every training and test fold (all schemes): {R['all_classes_in_every_fold']}\n")


def table_overall(folds, models, title):
    L.append(f"## {title}\n\n| Model | Precision (macro) | Recall (macro) | F1 (macro) | Accuracy | MCC | Weighted F1 |"
             "\n|---|---|---|---|---|---|---|")
    out = {}
    for m in models:
        g = lambda k: [f["metrics"][m][k] for f in folds]
        out[m] = {k: [float(np.mean(g(k))), float(np.std(g(k)))] for k in
                  ["macro_precision", "macro_recall", "macro_f1", "accuracy", "mcc", "weighted_f1",
                   "balanced_accuracy"]}
        out[m]["per_fold_mcc"] = g("mcc"); out[m]["per_fold_macro_f1"] = g("macro_f1")
        L.append(f"| {m} | {ms(g('macro_precision'))} | {ms(g('macro_recall'))} | {ms(g('macro_f1'))} | "
                 f"{ms(g('accuracy'))} | {ms(g('mcc'))} | {ms(g('weighted_f1'))} |")
    L.append("")
    return out


ALL = ["TabPFN-3"] + BASELINES
STUD = ["Student XGBoost", "Student Decision Tree", "Student MLP", "Student Transformer"]
STUD_ALL = STUD + ["Student Decision Tree (depth 4)"]
R["table6_5fold"] = table_overall(F5, ALL, "Table 6: patient-level 5-fold (mean ± std over folds)")
R["table9_5fold"] = table_overall(F5, ["TabPFN-3"] + STUD_ALL, "Table 9: distillation, patient-level 5-fold")
R["rep_overall"] = table_overall(REP, ALL + STUD_ALL + ["MLP hard labels only (alpha=0)",
                                                   "MLP soft labels only (alpha=1)"],
                                 "Repeated 3x5 (15 folds): overall (mean ± std)")


def per_class_table(folds, model, title):
    L.append(f"## {title}\n\n| Class | Precision | Recall (sensitivity) | Specificity | F1 | per-fold F1 |"
             "\n|---|---|---|---|---|---|")
    out = {}
    for c in CLASS_NAMES:
        g = lambda k: [f["metrics"][model]["per_class"][c][k] for f in folds]
        out[c] = {k: [float(np.nanmean(g(k))), float(np.nanstd(g(k)))] for k in
                  ["precision", "recall", "specificity", "f1"]}
        out[c]["per_fold_f1"] = [round(v, 4) for v in g("f1")]
        L.append(f"| {NICE[c]} | {ms(g('precision'))} | {ms(g('recall'))} | {ms(g('specificity'))} | "
                 f"{ms(g('f1'))} | {', '.join(f'{v:.3f}' for v in g('f1'))} |")
    L.append("")
    return out


R["table7_tabpfn_per_class"] = per_class_table(F5, "TabPFN-3", "Table 7 / A3: TabPFN-3 per class, 5-fold")
R["student_mlp_per_class"] = per_class_table(F5, "Student MLP", "Student MLP per class, 5-fold")

# --- patient-level evaluation + bootstrap (5-fold OOF) ---
oof = pd.DataFrame({"idx": np.concatenate([f["test_idx"] for f in F5])})
oof["y"] = y_all[oof.idx]; oof["patient"] = groups[oof.idx]
oof["fold"] = np.concatenate([[k + 1] * len(f["test_idx"]) for k, f in enumerate(F5)])
for m in ["TabPFN-3", "Student MLP", "XGBoost"]:
    oof["pred_" + m] = np.concatenate([f["pred"][m] for f in F5])
P5 = np.vstack([f["proba_tabpfn"] for f in F5])
for c in range(K):
    oof[f"p_{CLASS_NAMES[c]}"] = P5[:, c]
oof.to_csv(os.path.join(OUT, "oof_predictions_5fold.csv"), index=False)


def patient_level(o):
    g = o.groupby("patient")
    yp = g["y"].first().values
    pp = np.argmax(g[[f"p_{c}" for c in CLASS_NAMES]].mean().values, 1)
    return yp, pp


def summary_stats(o):
    y, p = o["y"].values, o["pred_TabPFN-3"].values
    py, pp = patient_level(o)
    cm = confusion_matrix(py, pp, labels=list(range(K)))
    d = {"cycle_macro_f1": f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0),
         "cycle_mcc": matthews_corrcoef(y, p), "cycle_balanced_acc": balanced_accuracy_score(y, p),
         "patient_accuracy": accuracy_score(py, pp), "patient_balanced_acc": balanced_accuracy_score(py, pp),
         "patient_macro_f1": f1_score(py, pp, average="macro", labels=list(range(K)), zero_division=0),
         "patient_mcc": matthews_corrcoef(py, pp)}
    for c in range(K):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        d[f"patient_sens_{CLASS_NAMES[c]}"] = tp / (tp + fn) if tp + fn else np.nan
        d[f"patient_spec_{CLASS_NAMES[c]}"] = tn / (tn + fp) if tn + fp else np.nan
        ycm = confusion_matrix(y, p, labels=list(range(K)))
        tp2 = ycm[c, c]; fn2 = ycm[c].sum() - tp2; fp2 = ycm[:, c].sum() - tp2; tn2 = ycm.sum() - tp2 - fn2 - fp2
        d[f"cycle_sens_{CLASS_NAMES[c]}"] = tp2 / (tp2 + fn2) if tp2 + fn2 else np.nan
        d[f"cycle_spec_{CLASS_NAMES[c]}"] = tn2 / (tn2 + fp2) if tn2 + fp2 else np.nan
    return d, cm


point, pcm = summary_stats(oof)
rng = np.random.RandomState(7)
pat_cls = oof.groupby("patient")["y"].first()
by_pat = {pt: g for pt, g in oof.groupby("patient")}
boots = []
for b in range(N_BOOT):
    pts = []
    for c in range(K):
        cand = pat_cls[pat_cls == c].index.to_numpy()
        pts += list(rng.choice(cand, len(cand), replace=True))
    parts = []
    for j, pt in enumerate(pts):
        gdf = by_pat[pt].copy(); gdf["patient"] = f"{pt}_{j}"; parts.append(gdf)
    boots.append(summary_stats(pd.concat(parts))[0])
bdf = pd.DataFrame(boots)
R["patient_level_and_ci"] = {k: {"value": float(point[k]), "ci95": [float(np.nanpercentile(bdf[k], 2.5)),
                                                                     float(np.nanpercentile(bdf[k], 97.5))]}
                             for k in point}
R["patient_confusion_matrix"] = pcm.tolist()
L.append("## Table A3: TabPFN-3 sensitivity / specificity (pooled 5-fold OOF), 95% patient-bootstrap CI\n")
L.append("| Class | Sens. (cycle) | Spec. (cycle) | Sens. (patient) | Spec. (patient) |\n|---|---|---|---|---|")
ci = R["patient_level_and_ci"]
f_ = lambda k: f"{ci[k]['value']:.3f} [{ci[k]['ci95'][0]:.3f}, {ci[k]['ci95'][1]:.3f}]"
for c in CLASS_NAMES:
    L.append(f"| {NICE[c]} | {f_('cycle_sens_'+c)} | {f_('cycle_spec_'+c)} | {f_('patient_sens_'+c)} | "
             f"{f_('patient_spec_'+c)} |")
L.append("\n| Summary | value [95% CI] |\n|---|---|")
for k in ["cycle_macro_f1", "cycle_mcc", "cycle_balanced_acc", "patient_accuracy", "patient_balanced_acc",
          "patient_macro_f1", "patient_mcc"]:
    L.append(f"| {k} | {f_(k)} |")
L.append(f"\nPatient-level confusion matrix (rows true, cols pred; order {CLASS_NAMES}):\n\n```\n{pcm}\n```\n")

# --- Table 8: Nadeau-Bengio TabPFN vs baselines (15 folds) ---
for metric in ["mcc", "macro_f1"]:
    L.append(f"## Table 8 ({metric}): TabPFN-3 vs baselines, Nadeau-Bengio, 15 folds, rho={RATIO:.4f}\n")
    L.append("| Baseline | Mean delta | t | p | Holm p_adj |\n|---|---|---|---|---|")
    a = [f["metrics"]["TabPFN-3"][metric] for f in REP]
    rows = [(m,) + nb_test(a, [f["metrics"][m][metric] for f in REP], RATIO)[:3] for m in BASELINES]
    adj = holm([r[3] for r in rows])
    R[f"table8_{metric}"] = []
    for r, pa in zip(rows, adj):
        R[f"table8_{metric}"].append({"baseline": r[0], "delta": r[1], "t": r[2], "p": r[3], "p_holm": pa})
        L.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.5f} | {pa:.5f} |")
    L.append("")

# --- Table 11 + TOST + ablation ---
for metric in ["mcc", "macro_f1", "macro_recall"]:
    L.append(f"## Table 11 ({metric}): teacher vs students, Nadeau-Bengio, 15 folds\n")
    L.append("| Student | Mean loss | t | p | Holm p_adj |\n|---|---|---|---|---|")
    a = [f["metrics"]["TabPFN-3"][metric] for f in REP]
    rows = [(m,) + nb_test(a, [f["metrics"][m][metric] for f in REP], RATIO)[:3] for m in STUD_ALL]
    adj = holm([r[3] for r in rows])
    R[f"table11_{metric}"] = []
    for r, pa in zip(rows, adj):
        R[f"table11_{metric}"].append({"student": r[0], "delta": r[1], "t": r[2], "p": r[3], "p_holm": pa})
        L.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.5f} | {pa:.5f} |")
    L.append("")
R["tost_mcc"] = {}
L.append(f"## TOST on MCC (teacher - student), margin {EQ_MARGIN}, NB-corrected 90% CI\n\n"
         "| Student | Mean diff | 90% CI | p_TOST | Equivalent |\n|---|---|---|---|---|")
for m in ["Student MLP", "Student Transformer"]:
    tt = tost([f["metrics"]["TabPFN-3"]["mcc"] for f in REP], [f["metrics"][m]["mcc"] for f in REP], RATIO,
              EQ_MARGIN)
    R["tost_mcc"][m] = tt
    L.append(f"| {m} | {tt['mean_diff']:.4f} | [{tt['ci90_low']:.4f}, {tt['ci90_high']:.4f}] | "
             f"{tt['p_tost']:.4f} | {tt['equivalent']} |")
L.append("")
L.append("## Table A4: hard vs soft vs combined (Student MLP, 15 folds)\n\n| Target | Accuracy | MCC | Macro F1 | "
         "Macro Recall |\n|---|---|---|---|---|")
R["table_a4"] = {}
for m, lab in [("MLP hard labels only (alpha=0)", "Hard labels only (alpha = 0)"),
               ("MLP soft labels only (alpha=1)", "Soft labels only (alpha = 1)"),
               ("Student MLP", "Combined loss (alpha = 0.5)")]:
    g = lambda k: [f["metrics"][m][k] for f in REP]
    R["table_a4"][lab] = {k: [float(np.mean(g(k))), float(np.std(g(k)))] for k in
                          ["accuracy", "mcc", "macro_f1", "macro_recall"]}
    L.append(f"| {lab} | {ms(g('accuracy'))} | {ms(g('mcc'))} | {ms(g('macro_f1'))} | {ms(g('macro_recall'))} |")
for other in ["MLP hard labels only (alpha=0)", "MLP soft labels only (alpha=1)"]:
    d, t, p, _ = nb_test([f["metrics"]["Student MLP"]["mcc"] for f in REP],
                         [f["metrics"][other]["mcc"] for f in REP], RATIO)
    R["table_a4"][f"combined_vs_{other}"] = {"delta_mcc": d, "t": t, "p": p}
    L.append(f"\nCombined vs {other}: delta MCC = {d:.4f}, t = {t:.3f}, p = {p:.4f}")
L.append("")

# --- Table 1: group permutation importance (15 folds) ---
L.append("## Table 1: feature-group permutation importance, TabPFN-3, 15 folds\n\n| Group | N_g | Total weight | "
         "Per-feature weight | Mean accuracy drop | Mean MCC drop |\n|---|---|---|---|---|---|")
R["table1"] = {}
for g, cols in GROUPS.items():
    tw = [f["perm_norm"][g] for f in REP]; pw = [v / len(cols) for v in tw]
    da = [f["perm_acc"][g] for f in REP]; dm = [f["perm_mcc"][g] for f in REP]
    R["table1"][g] = {"n": len(cols), "total": [float(np.mean(tw)), float(np.std(tw))],
                      "per_feature": [float(np.mean(pw)), float(np.std(pw))],
                      "acc_drop": [float(np.mean(da)), float(np.std(da))],
                      "mcc_drop": [float(np.mean(dm)), float(np.std(dm))]}
    L.append(f"| {g} | {len(cols)} | {ms(tw)} | {ms(pw)} | {ms(da)} | {ms(dm)} |")
L.append("")

# --- SHAP ---
SV = np.concatenate([f["shap_vals"] for f in F5]); SY = np.concatenate([f["shap_y"] for f in F5])
glob = np.abs(SV).mean(axis=(0, 2))
order = np.argsort(glob)[::-1]
fold_imp = np.array([np.abs(f["shap_vals"]).mean(axis=(0, 2)) for f in F5])
rhos = [stats.spearmanr(fold_imp[i], fold_imp[j])[0] for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS)]
top10 = [set(np.argsort(fi)[::-1][:10]) for fi in fold_imp]
overlap = [len(top10[i] & top10[j]) / 10 for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS)]
by_class = {}
for c in range(K):
    mk = SY == c
    if mk.any():
        imp = np.abs(SV[mk, :, c]).mean(0)
        by_class[NICE[CLASS_NAMES[c]]] = [(FEATS[i], float(imp[i])) for i in np.argsort(imp)[::-1][:5]]
R["shap"] = {"n_explained": int(len(SY)), "explainer": "shap.PermutationExplainer", "max_evals_per_sample":
             2 * (2 * 52) + 1, "background_per_fold": SHAP_BG, "output": "predicted class probabilities",
             "global_top20": [(FEATS[i], float(glob[i])) for i in order[:20]],
             "fold_rank_spearman_mean": float(np.nanmean(rhos)), "fold_rank_spearman_min": float(np.nanmin(rhos)),
             "top10_overlap_mean": float(np.mean(overlap)), "by_class_top5": by_class}
L.append(f"## SHAP (TabPFN-3, {len(SY)} explained test cycles over 5 folds)\n")
L.append("Top 20 (mean |SHAP| over samples and classes):\n")
for i in order[:20]:
    L.append(f"- {FEATS[i]}: {glob[i]:.4f}")
L.append(f"\nFold stability: mean pairwise Spearman = {np.nanmean(rhos):.3f} (min {np.nanmin(rhos):.3f}); "
         f"mean top-10 overlap = {np.mean(overlap):.2f}\n\nClass-specific top 5:\n")
for c, lst in by_class.items():
    L.append(f"- {c}: " + ", ".join(f"{a} ({b:.4f})" for a, b in lst))
L.append("")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    pretty = lambda s: (s.replace("mfcc_mean_", "MFCC mean ").replace("mfcc_std_", "MFCC std ")
                        .replace("spectral_", "Spectral ").replace("_mean", " (mean)").replace("_std", " (std)")
                        .replace("zcr", "ZCR").replace("rms", "RMS").replace("crackle", "Crackle")
                        .replace("wheeze", "Wheeze"))
    top = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.barh([pretty(FEATS[i]) for i in top], glob[top], color="#5B75A6")
    ax.set_xlabel("Mean |SHAP value| (averaged over test cycles and classes)")
    ax.xaxis.grid(True, ls="--", alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_shap_global.png")); plt.close()
    feats10 = order[:10]
    mat = np.array([[np.abs(SV[SY == c][:, i, c]).mean() if (SY == c).any() else np.nan for i in feats10]
                    for c in range(K)])
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels([pretty(FEATS[i]) for i in feats10], rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels([NICE[c] for c in CLASS_NAMES])
    plt.colorbar(im, label="Mean |SHAP| for own class"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figA_shap_by_class.png")); plt.close()
except Exception as e:
    log("plot error", e)

# --- rules ---
R["table10_rules"] = rep_rules
R["table10_tree_fidelity_to_teacher"] = R_TREE_FIDELITY
L.append("## Table 10: rules from the distilled depth-4 decision tree (pooled data)\n\n"
         "| Class | Rule | Confidence | Support | Lift |\n|---|---|---|---|---|")
for r in rep_rules:
    L.append(f"| {r['class']} | {r['rule']} | {100*r['confidence']:.1f}% | {r['support']} | {r['lift']:.2f} |")
L.append(f"\nAgreement of the depth-4 tree with the teacher (argmax, pooled): {R_TREE_FIDELITY:.3f}")
L.append(f"\n(min support {MIN_SUP}; all leaves in student_tree_all_leaves.csv; full tree in student_tree_full.txt)\n")

L.append(f"Student parameters: {R['info']['student_params']}\n")
with open(os.path.join(OUT, "summary.md"), "w") as fh:
    fh.write("\n".join(L))


def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=_js)
log(f"DONE. Results in {OUT}/summary.md and {OUT}/results.json")
