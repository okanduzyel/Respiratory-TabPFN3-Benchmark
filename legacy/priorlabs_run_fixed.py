import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import tabpfn_client
from tabpfn_client import TabPFNClassifier, set_access_token
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    accuracy_score, f1_score
)

warnings.filterwarnings("ignore")

# macOS OpenMP / Thread çakışmasını engellemek için çevre değişkenleri kilitleniyor
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==== Authenticate ====
API_TOKEN = os.environ.get("PRIORLABS_TOKEN", "")  # set the key as an environment variable  # Kendi API key'iniz
tabpfn_client.set_access_token(API_TOKEN)

# ==== AYARLAR ====
INPUT_CSV = "respiratory_dataset.csv"
TARGET_COL = "label_id"
# 52 Feature yapısı için crackle ve wheeze DROP_COLS içinden çıkarıldı
DROP_COLS = ["filename", "patient_id", "label", "split"]
N_FOLDS = 5
N_REPEATS = 3  # 3x5 = 15 fold elde etmek için tekrar sayısı kilitlendi

MODEL_NAME = "TabPFN"
RESULTS_CSV = "fold_scores_for_nb_test.csv"

# Sıfırlama Garantisi: Eski çalıştırmalardan kalan CSV dosyasını temizleyelim
if os.path.exists(RESULTS_CSV):
    os.remove(RESULTS_CSV)

CLASSICAL_MODELS = {
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "MLP": lambda: MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
    "RandomForest": lambda: RandomForestClassifier(random_state=42, n_estimators=300, class_weight="balanced"),
    "SVM": lambda: SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced"),
    "NaiveBayes": lambda: GaussianNB(),
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "XGBoost": lambda: XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=1, tree_method="hist"),
}

label_mapping = {
    'bronchiectasis': 0, 'bronchiolitis': 1, 'copd': 2,
    'healthy': 3, 'pneumonia': 4, 'urti': 5
}
class_labels = list(label_mapping.keys())

# ==== Veri Yükleme ====
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

X_all = df.drop(columns=DROP_COLS + [TARGET_COL], errors='ignore')
y_all = df[TARGET_COL].to_numpy()
patient_ids = df["patient_id"].to_numpy()

print(f"Toplam {len(df)} satır bulundu.")
print(f"Benzersiz hasta sayısı: {len(np.unique(patient_ids))}")
print(f"Kullanılan Özellik (Feature) Sayısı: {X_all.shape[1]}")

detailed_reports = {}

def save_fold_scores_for_nb_test(model_name, accs, mccs, f1s, stage_label, csv_path=RESULTS_CSV):
    """ 15 fold'un tamamını CSV dosyasına ekleyerek biriktirir """
    fold_results_df = pd.DataFrame({
        "model": [model_name] * len(accs),
        "stage": [stage_label] * len(accs),
        "fold": list(range(1, len(accs) + 1)),
        "accuracy": accs,
        "mcc": mccs,
        "macro_f1": f1s,
    })
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        fold_results_df = pd.concat([existing, fold_results_df], ignore_index=True)
    fold_results_df.to_csv(csv_path, index=False)


# ==============================================================================
# ==== HIZLI VE SIZINTISIZ GRUP PERMÜTASYON HESAPLAYICI ====
# ==============================================================================
feature_cols = list(X_all.columns)
feature_group_indices = {
    "Adventitious Sounds": [i for i, c in enumerate(feature_cols) if c in ['crackle', 'wheeze']],
    "Spectral Cepstral (Mean)": [i for i, c in enumerate(feature_cols) if c.startswith('mfcc_mean')],
    "Spectral Cepstral (Std)": [i for i, c in enumerate(feature_cols) if c.startswith('mfcc_std')],
    "Temporal Domain Metrics": [i for i, c in enumerate(feature_cols) if c.startswith('zcr') or c.startswith('rms')],
    "Spectral Domain Metrics": [i for i, c in enumerate(feature_cols) if c.startswith('spectral_')]
}
feature_group_indices = {k: v for k, v in feature_group_indices.items() if len(v) > 0}

fold_group_totals = {g: [] for g in feature_group_indices}
fold_per_feature_means = {g: [] for g in feature_group_indices}

def calculate_fast_group_importance(model, X_test, y_test, seed=42):
    """ API istek sayısını 260'tan 5'e düşüren hızlı grup permütasyonu """
    np.random.seed(seed)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    group_drops = {}

    X_test_arr = X_test.values
    for group_name, idx_list in feature_group_indices.items():
        X_permuted = X_test_arr.copy()
        for idx in idx_list:
            X_permuted[:, idx] = np.random.permutation(X_permuted[:, idx])
        
        perm_pred = model.predict(X_permuted)
        perm_acc = accuracy_score(y_test, perm_pred)
        group_drops[group_name] = max(0.0, baseline_acc - perm_acc)

    total_drop = sum(group_drops.values())
    if total_drop == 0:
        total_drop = 1.0

    group_weights = {g: v / total_drop for g, v in group_drops.items()}
    return group_weights


# ==============================================================================
# ==== STAGE 2: 3x5 FOLD CROSS-VALIDATION (HASTA BAZLI / TABPFN3) ====
# ==============================================================================
print(f"\n\n{'#'*75}\n{' '*5}STAGE 2: 3x5 REPEATED PATIENT-LEVEL VALIDATION (15 FOLDS)\n{'#'*75}")

tabpfn_accs, tabpfn_mccs, tabpfn_f1s, tabpfn_reports = [], [], [], []
global_fold_counter = 1

for repeat in range(N_REPEATS):
    current_seed = 42 + repeat
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=current_seed)
    
    for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=patient_ids), 1):
        X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
        X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]

        print(f"🤖 TabPFN3 -> Global Fold {global_fold_counter}/15 (Repeat {repeat+1}, Fold {fold_num}/5) çalıştırılıyor...")
        
        model = TabPFNClassifier(n_estimators=8)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tabpfn_accs.append(accuracy_score(y_test, y_pred))
        tabpfn_mccs.append(matthews_corrcoef(y_test, y_pred))
        tabpfn_f1s.append(f1_score(y_test, y_pred, average="macro"))

        # --- HIZLI GRUP BAZLI PREDICTION ---
        g_weights = calculate_fast_group_importance(model, X_test, y_test, seed=current_seed)
        for group_name, idx_list in feature_group_indices.items():
            g_count = len(idx_list)
            g_total_weight = g_weights[group_name]
            g_per_feat_weight = g_total_weight / g_count
            fold_group_totals[group_name].append(g_total_weight)
            fold_per_feature_means[group_name].append(g_per_feat_weight)
        
        report = classification_report(
            y_test, y_pred, labels=list(label_mapping.values()),
            target_names=class_labels, digits=4, output_dict=True, zero_division=0
        )
        tabpfn_reports.append(report)
        global_fold_counter += 1

save_fold_scores_for_nb_test(MODEL_NAME, tabpfn_accs, tabpfn_mccs, tabpfn_f1s, stage_label="patient_level")
detailed_reports[MODEL_NAME] = {
    "accs": tabpfn_accs, "mccs": tabpfn_mccs, "reports": tabpfn_reports
}


# ==============================================================================
# ==== 3x5 FOLD CROSS-VALIDATION (HASTA BAZLI / KLASİK MODELLER) ====
# ==============================================================================
for clf_name, clf_factory in CLASSICAL_MODELS.items():
    print(f"\n\n{'#'*75}\n{' '*5}⚡ {clf_name} MİMARİSİ İÇİN 15 FOLD EĞİTİMİ BAŞLADI\n{'#'*75}")
    clf_accs, clf_mccs, clf_f1s, clf_reports = [], [], [], []
    global_fold_counter = 1

    for repeat in range(N_REPEATS):
        current_seed = 42 + repeat
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=current_seed)
        
        for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=patient_ids), 1):
            X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
            X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = clf_factory()
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            clf_accs.append(accuracy_score(y_test, y_pred))
            clf_mccs.append(matthews_corrcoef(y_test, y_pred))
            clf_f1s.append(f1_score(y_test, y_pred, average="macro"))
            
            report = classification_report(
                y_test, y_pred, labels=list(label_mapping.values()),
                target_names=class_labels, digits=4, output_dict=True, zero_division=0
            )
            clf_reports.append(report)
            global_fold_counter += 1

    save_fold_scores_for_nb_test(clf_name, clf_accs, clf_mccs, clf_f1s, stage_label="patient_level")
    detailed_reports[clf_name] = {
        "accs": clf_accs, "mccs": clf_mccs, "reports": clf_reports
    }


# ==============================================================================
# ==== 15 FOLD NADEAU-BENGIO MODEL KARŞILAŞTIRMA MATRİSİ ====
# ==============================================================================
print(f"\n\n{'='*95}\n{' '*15}🚨 NADEAU-BENGIO REPEATED TEST MATRIX (15 FOLDS vs. TabPFN3)\n{'='*95}")

def nadeau_bengio_test(scores_target, scores_baseline, n_folds=5, n_repeats=3):
    diff = np.array(scores_target) - np.array(scores_baseline)
    n_total = n_folds * n_repeats  # 15
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)
    if var_diff == 0:
        return mean_diff, (np.inf if mean_diff != 0 else 0.0), (0.0 if mean_diff != 0 else 1.0)
    
    correction = (1 / n_total) + 0.25
    denom = np.sqrt(correction * var_diff)
    t_stat = mean_diff / denom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_total - 1))
    return mean_diff, t_stat, p_value

results_df = pd.read_csv(RESULTS_CSV)
stage_df = results_df[results_df["stage"] == "patient_level"]
metrics_to_test = ["mcc", "accuracy", "macro_f1"]

target_model = "TabPFN"
other_models = [m for m in sorted(stage_df["model"].unique()) if m != target_model]

for metric in metrics_to_test:
    print(f"\n--- Evaluation Metric: {metric.upper()} (15 Folds Matrix) ---")
    header_line = f"{'Target Model (A)':18s}{'Baseline Model (B)':22s}{'Mean Delta (A-B)':>16s}{'t-Statistic':>14s}{'p-value':>12s}{'Significant':>12s}"
    print(header_line)
    print("-" * len(header_line))
    
    scores_target = stage_df[stage_df["model"] == target_model][metric].values
    
    for baseline_model in other_models:
        scores_baseline = stage_df[stage_df["model"] == baseline_model][metric].values
        
        if len(scores_target) != len(scores_baseline) or len(scores_target) == 0:
            print(f"Hata: {baseline_model} için fold boyutu eşleşmedi ({len(scores_baseline)})!")
            continue
            
        mean_diff, t_stat, p_value = nadeau_bengio_test(scores_target, scores_baseline, n_folds=N_FOLDS, n_repeats=N_REPEATS)
        
        if p_value < 0.05 and mean_diff > 0:
            decision = "YES (A Wins)"
        elif p_value < 0.05 and mean_diff < 0:
            decision = "YES (B Wins)"
        else:
            decision = "NO (Parity)"
            
        p_str = f"{p_value:.5f}" if p_value >= 0.001 else "< 0.001"
        print(f"{target_model+'3':18s}{baseline_model:22s}{mean_diff:16.4f}{t_stat:14.4f}{p_str:>12s}{decision:>12s}")

print(f"{'='*95}\n")

# 15 Fold Genel Ortalama Özeti
summary = stage_df.groupby("model")[["mcc", "accuracy", "macro_f1"]].agg(["mean", "std"])
print("--- Overall 15-Fold Rankings (Mean ± Std) ---")
print(summary)


# ==============================================================================
# ==== OTOMATİK HESAPLANAN REVISED TABLE 1 ÇIKTISI ====
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"📊 REVISED TABLE 1 ({X_all.shape[1]} FEATURES / 15-FOLD TABPFN-3 CALCULATED RESULTS)")
print(f"{'='*100}")

table1_rows = []
for group_name, idx_list in feature_group_indices.items():
    g_count = len(idx_list)
    total_w_mean = np.mean(fold_group_totals[group_name])
    total_w_std = np.std(fold_group_totals[group_name])
    per_feat_mean = np.mean(fold_per_feature_means[group_name])
    per_feat_std = np.std(fold_per_feature_means[group_name])
    
    table1_rows.append({
        "Feature Group": group_name,
        "Dimensionality (Ng)": g_count,
        "Total Group Weight (Sum Wg)": f"{total_w_mean:.4f} ± {total_w_std:.4f}",
        "Per-Feature Normalized Weight (W_feat)": f"{per_feat_mean:.4f} ± {per_feat_std:.4f}"
    })

df_table1_final = pd.DataFrame(table1_rows)
print(df_table1_final.to_string(index=False))
print(f"{'='*100}\n")


# ==============================================================================
# ==== DETAYLI 15-FOLD DETAILED CLASSIFICATION REPORT (MEAN ± STD + MCC) ====
# ==============================================================================
print(f"\n\n{'='*105}\n{' '*15}📊 DETAILED CLASSIFICATION REPORTS ACROSS ALL FOLDS (MEAN ± STD)\n{'='*105}")

row_keys = class_labels + ["macro avg", "weighted avg"]
metric_keys = ["precision", "recall", "f1-score"]

def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"

col_w = 17

for model_name, data in detailed_reports.items():
    print(f"\n{'='*85}")
    display_model_name = "TabPFN3" if model_name == "TabPFN" else model_name
    print(f"📋 Model: {display_model_name}")
    print(f"Accuracy : {np.mean(data['accs']):.4f} ± {np.std(data['accs']):.4f}")
    print(f"MCC      : {np.mean(data['mccs']):.4f} ± {np.std(data['mccs']):.4f}")
    print(f"{'='*85}")

    collected = {row: {m: [] for m in metric_keys} for row in row_keys}
    fold_supports = {row: [] for row in class_labels}

    for report in data["reports"]:
        for row in row_keys:
            if row not in report:
                continue
            for m in metric_keys:
                collected[row][m].append(report[row][m])
        for label in class_labels:
            if label in report:
                fold_supports[label].append(report[label]["support"])

    header = f"{'Label / Metric':22s}{'precision':>{col_w}}{'recall':>{col_w}}{'f1-score':>{col_w}}{'support':>{col_w}}"
    print(header)
    print("-" * (22 + col_w * 4))

    for row in row_keys:
        if row not in collected or not collected[row]["precision"]:
            continue
        line = f"{row:22s}"
        for m in metric_keys:
            v = collected[row][m]
            cell = fmt(np.mean(v), np.std(v))
            line += f"{cell:>{col_w}}"

        if row in fold_supports:
            sup = int(np.mean(fold_supports[row]))
            line += f"{sup:>{col_w}}"
        else:
            total_sup = int(sum(np.mean(fold_supports[lbl]) for lbl in class_labels))
            line += f"{total_sup:>{col_w}}"

        print(line)
    print()
