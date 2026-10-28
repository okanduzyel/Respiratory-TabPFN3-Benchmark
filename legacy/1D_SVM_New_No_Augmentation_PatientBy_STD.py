import os
import numpy as np
import librosa
import random
import re
import joblib

# ==== Kütüphaneler ====
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import warnings

warnings.filterwarnings("ignore")

# ==== AYARLAR ====
# 🚀 GÜNCELLEME: LRTI sınıfı çıkarıldı ve etiketler sıfırdan boşluksuz olacak şekilde yeniden numaralandırıldı.
label_mapping = {
    'bronchiectasis': 0,
    'bronchiolitis': 1,
    'copd': 2,
    'healthy': 3,
    'pneumonia': 4,
    'urti': 5
}
class_labels = list(label_mapping.keys())
data_dir = "/Users/okanduzyel/Downloads/Last_ICHBI2"
N_FOLDS = 5
SAMPLE_RATE = 22050

# === Özellik Çıkarımı ===
def extract_features(y, sr, crackle, wheeze):
    feats = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(np.mean(zcr))
    feats.append(np.std(zcr))
    rms = librosa.feature.rms(y=y)
    feats.append(np.mean(rms))
    feats.append(np.std(rms))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.append(np.mean(centroid))
    feats.append(np.std(centroid))
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats.append(np.mean(bandwidth))
    feats.append(np.std(bandwidth))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.append(np.mean(rolloff))
    feats.append(np.std(rolloff))
    feats.append(crackle)
    feats.append(wheeze)
    return np.array(feats)

def parse_crackle_wheeze_from_filename(filename):
    match = re.search(r'_cr([01])_wh([01])', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

# ==== Fold bazında mean ± std yazdıran yardımcı fonksiyon ====
def print_metrics_with_std(model_name, fold_accs, fold_mccs, fold_reports):
    print(f"\n{model_name} - 5 Fold Genel Sonuçlar (mean ± std)")
    print(f"Accuracy : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"MCC      : {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}\n")

    row_keys = class_labels + ["macro avg", "weighted avg"]
    metric_keys = ["precision", "recall", "f1-score"]

    collected = {row: {m: [] for m in metric_keys} for row in row_keys}
    fold_supports = {row: [] for row in class_labels}

    for report in fold_reports:
        for row in row_keys:
            if row not in report:
                continue
            for m in metric_keys:
                collected[row][m].append(report[row][m])
        for label in class_labels:
            if label in report:
                fold_supports[label].append(report[label]["support"])

    col_w = 13
    header = f"{'':20s}" + "".join(f"{'precision':>{col_w}}{'':3s}") \
                         + "".join(f"{'recall':>{col_w}}{'':3s}") \
                         + "".join(f"{'f1-score':>{col_w}}{'':3s}") \
                         + f"{'support':>{col_w}}"
    print(header)
    print()

    def fmt(mean, std):
        return f"{mean:.4f}±{std:.4f}"

    for row in row_keys:
        if row not in collected:
            continue
        vals = collected[row]
        if not vals["precision"]:
            continue

        line = f"{row:20s}"
        for m in metric_keys:
            v = vals[m]
            cell = fmt(np.mean(v), np.std(v))
            line += f"{cell:>{col_w+3}}"

        if row in fold_supports:
            sup = int(np.sum(fold_supports[row]))
            line += f"{sup:>{col_w}}"
        else:
            total_sup = int(sum(np.sum(fold_supports[lbl]) for lbl in class_labels))
            line += f"{total_sup:>{col_w}}"

        print(line)

    print()

# === Veri Yükleme ===
filepaths, labels, crackles, wheezes, patient_ids = [], [], [], [], []

for fold_name in sorted(os.listdir(data_dir)):
    fold_dir = os.path.join(data_dir, fold_name)
    if not os.path.isdir(fold_dir): continue
    for class_name in sorted(os.listdir(fold_dir)):
        # 🚀 GÜNCELLEME: Asthma ve LRTI sınıfları okuma aşamasında tamamen es geçiliyor.
        if class_name.lower() in ["asthma", "lrti"]:
            continue
        class_dir = os.path.join(fold_dir, class_name)
        if not os.path.isdir(class_dir): continue
        if class_name.lower() not in label_mapping:
            print(f"Uyarı: {class_name} label_mapping'te yok, atlanıyor.")
            continue
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith('.wav'):
                path = os.path.join(class_dir, filename)
                filepaths.append(path)
                labels.append(label_mapping[class_name.lower()])
                
                cr, wh = parse_crackle_wheeze_from_filename(filename)
                crackles.append(cr)
                wheezes.append(wh)
                
                patient_id = filename.split('_')[0]
                patient_ids.append(patient_id)

filepaths = np.array(filepaths)
labels = np.array(labels)
crackles = np.array(crackles)
wheezes = np.array(wheezes)
patient_ids = np.array(patient_ids)

print(f"Toplam {len(filepaths)} ses dosyası bulundu.")
print(f"Benzersiz hasta sayısı: {len(np.unique(patient_ids))}")

# ==== Özellik çıkarımı ====
X_all, y_all = [], []
for i, path in enumerate(filepaths):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    cr, wh = crackles[i], wheezes[i]
    X_all.append(extract_features(y, sr, cr, wh))
    y_all.append(labels[i])
X_all = np.array(X_all)
y_all = np.array(y_all)

# ==== SVM GRID SEARCH PARAMETRE SETİ ====
param_grid = [
    {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 1.0, "gamma": "auto"},
    {"kernel": "rbf", "C": 10.0, "gamma": "auto"},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.01},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.01},
    {"kernel": "linear", "C": 0.1, "gamma": "scale"},
    {"kernel": "linear", "C": 1.0, "gamma": "scale"},
    {"kernel": "poly", "C": 1.0, "gamma": "scale"},
]

# ==== 5-Fold Cross-Validation (HASTA BAZLI / PATIENT-LEVEL) — SVM Grid Search ====
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

grid_summary = []

for params in param_grid:
    model_name = f"SVM (kernel={params['kernel']}, C={params['C']}, gamma={params['gamma']})"
    print(f"\n{'='*60}\nEğitiliyor ve Test Ediliyor: {model_name}\n{'='*60}")

    clf = SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        class_weight="balanced",
        random_state=42
    )

    all_true, all_pred = [], []
    fold_accs, fold_mccs, fold_reports = [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=patient_ids), 1):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # SVM mesafe tabanlı olduğu için her fold'da ölçeklendirme yapılır
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)

        # Modeli eğit
        clf.fit(X_train_fit, y_train)

        # Test et
        y_pred = clf.predict(X_test_fit)

        # Fold bazında metrikleri topla
        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
        fold_reports.append(classification_report(
            y_test, y_pred,
            labels=list(label_mapping.values()),
            target_names=class_labels,
            digits=4,
            output_dict=True,
            zero_division=0
        ))

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    # Sonuçları Yazdır
    acc = accuracy_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)

    print(f"{model_name} - 5 Fold Genel Sonuçlar (Hasta Bazlı Bölme)")
    print(f"Accuracy : {acc:.4f}")
    print(f"MCC      : {mcc:.4f}\n")
    print(classification_report(all_true, all_pred, target_names=class_labels, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_true, all_pred))

    # mean ± std çıktısı
    print_metrics_with_std(model_name, fold_accs, fold_mccs, fold_reports)

    grid_summary.append({
        "params": params,
        "accuracy": acc,
        "mcc": mcc,
        "macro_f1": np.mean([r["macro avg"]["f1-score"] for r in fold_reports])
    })

# ==== GRID SEARCH ÖZET TABLOSU ====
print(f"\n{'='*60}\nGRID SEARCH ÖZET (MCC'ye göre sıralı)\n{'='*60}")
grid_summary_sorted = sorted(grid_summary, key=lambda x: x["mcc"], reverse=True)
for i, res in enumerate(grid_summary_sorted, 1):
    p = res["params"]
    print(f"{i}. kernel={p['kernel']:<7} C={p['C']:<6} gamma={p['gamma']:<8} "
          f"-> Accuracy={res['accuracy']:.4f}  MCC={res['mcc']:.4f}  Macro-F1={res['macro_f1']:.4f}")
