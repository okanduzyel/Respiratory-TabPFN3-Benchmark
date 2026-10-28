import os
import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNClassifier, set_access_token
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    accuracy_score, f1_score
)

# ==== Geleneksel Modellerin Kütüphaneleri ====
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# ==== Authenticate ====
API_TOKEN = os.environ.get("PRIORLABS_TOKEN", "")  # set the key as an environment variable  # kendi API key'inle değiştir
tabpfn_client.set_access_token(API_TOKEN)

# ==== AYARLAR ====
INPUT_CSV = "respiratory_dataset_augmented.csv"
TARGET_COL = "label"
N_FOLDS = 5
label_mapping = {
    'bronchiectasis': 0, 'bronchiolitis': 1, 'copd': 2,
    'healthy': 3, 'pneumonia': 4, 'urti': 5
}
class_labels = list(label_mapping.keys())

# ==== Veri Yükleme ====
df = pd.read_csv(INPUT_CSV)

# Veri setindeki olası boşluk hatalarını standartlaştıralım
df.columns = df.columns.str.strip()

# Target kolonu içerisindeki string sınıfları label_mapping'e göre sayısal id'lere çevirelim
if df[TARGET_COL].dtype == object or df[TARGET_COL].dtype == str:
    df['label_id'] = df[TARGET_COL].map(label_mapping)
    actual_target_col = 'label_id'
else:
    actual_target_col = TARGET_COL

# Sadece feature olan kolonları ayıralım (label ve label_id hariç her şey feature'dır)
X_all = df.drop(columns=[TARGET_COL, 'label_id'], errors='ignore')
y_all = df[actual_target_col].to_numpy()

print(f"Toplam {len(df)} satır bulundu.")
print(f"Özellik (Feature) sayısı: {X_all.shape[1]}")

# ==============================================================================
# ==== MODELLERİN TANIMLANMASI (HAFIZA SIZINTISI VE ÇÖKME ENGELLEMELİ) ====
# ==============================================================================
models_dict = {
    "TabPFN3-Plus": TabPFNClassifier(n_estimators=8),
    
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=42
    ),
    
    "KNN": KNeighborsClassifier(
        metric="manhattan", n_neighbors=3, weights="distance", n_jobs=1
    ),
    
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=0.0001, learning_rate_init=0.001, random_state=42, max_iter=200
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,        # Bellek taşmasını önlemek için optimize edildi
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="entropy",
        bootstrap=False,
        random_state=42,
        n_jobs=1                 # Paralel thread patlamasını engeller
    ),
    
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale", random_state=42
    ),
    
    "Naive Bayes": GaussianNB(
        var_smoothing=1e-9
    ),
    
    "Logistic Regression": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=100, random_state=42, n_jobs=1
    )
}

# ==============================================================================
# ==== 5-FOLD CROSS-VALIDATION DÖNGÜSÜ (STAGE 1: CYCLE-LEVEL) ====
# ==============================================================================
print(f"\n\n{'#'*80}\n{' '*5}STAGE 1: CYCLE-LEVEL BENCHMARK EVALUATION (ALL MODELS)\n{'#'*80}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Tüm modellerin nihai özet tablosu için metrikleri toplayacağımız yapı
summary_results = {}

for model_name, model_obj in models_dict.items():
    print(f"\n🚀 {model_name} çalıştırılıyor...")
    
    fold_accs, fold_mccs, fold_f1s, fold_reports = [], [], [], []
    all_true, all_pred = [], []
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_train_c, y_train_c = X_all.iloc[train_idx], y_all[train_idx]
        X_test_c, y_test_c = X_all.iloc[test_idx], y_all[test_idx]
        
        # Modeli kopyalayıp eğitiyoruz
        from sklearn.base import clone
        clf = clone(model_obj) if model_name != "TabPFN3-Plus" else model_obj
        
        clf.fit(X_train_c, y_train_c)
        y_pred_c = clf.predict(X_test_c)
        
        # Metrik Hesaplamaları
        acc_c = accuracy_score(y_test_c, y_pred_c)
        mcc_c = matthews_corrcoef(y_test_c, y_pred_c)
        macro_f1_c = f1_score(y_test_c, y_pred_c, average="macro")
        
        fold_accs.append(acc_c)
        fold_mccs.append(mcc_c)
        fold_f1s.append(macro_f1_c)
        
        fold_reports.append(classification_report(
            y_test_c, y_pred_c,
            labels=list(label_mapping.values()),
            target_names=class_labels,
            digits=4,
            output_dict=True,
            zero_division=0
        ))
        
        all_true.extend(y_test_c.tolist())
        all_pred.extend(y_pred_c.tolist())
    
    # Modelin sonuçlarını summary_results içerisine kaydedelim
    summary_results[model_name] = {
        "acc_mean": np.mean(fold_accs), "acc_std": np.std(fold_accs),
        "mcc_mean": np.mean(fold_mccs), "mcc_std": np.std(fold_mccs),
        "f1_mean": np.mean(fold_f1s), "f1_std": np.std(fold_f1s),
        "fold_reports": fold_reports,
        "all_true": all_true, "all_pred": all_pred
    }
    
    # ==== Model Spesifik Detaylı Çıktılar ====
    print(f"\n📊 {model_name} - DETAYLI PERFORMANS RAPORU (4 BASAMAKLI):")
    print(f"--------------------------------------------------")
    print(f"Genel Doğruluk (Accuracy) : {accuracy_score(all_true, all_pred):.4f}")
    print(f"Genel Matthews Koralasyon Katsayısı (MCC) : {matthews_corrcoef(all_true, all_pred):.4f}\n")
    
    print(classification_report(
        all_true, all_pred,
        labels=list(label_mapping.values()),
        target_names=class_labels,
        digits=4,
        zero_division=0
    ))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_pred, labels=list(label_mapping.values())))
    print(f"--------------------------------------------------")
    print(f"✨ {model_name} Tamamlandı -> Mean Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f} | Mean MCC: {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}\n")

# ==============================================================================
# ==== NİHAİ AKADEMİK RAPORLAMA VE KIYASLAMA TABLOSU ====
# ==============================================================================
print(f"\n\n{'='*100}\n🏆 5-FOLD STAGE 1 NİHAİ PERFORMANS TABLOSU (CYCLE-LEVEL OVERVIEW)\n{'='*100}")

col_w = 18
header = f"{'Classifier':25s}{'Accuracy':>{col_w}}{'MCC':>{col_w}}{'Macro Precision':>{col_w}}{'Macro Recall':>{col_w}}{'Macro F1':>{col_w}}"
print(header)
print("-" * (25 + col_w * 5))

# DÜZELTME: Eksik olan formatlama fonksiyonunu geri ekledik
def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"

for model_name, res in summary_results.items():
    # Katmanlardan macro precision ve recall varyanslarını çekme
    f_precisions = [r["macro avg"]["precision"] for r in res["fold_reports"]]
    f_recalls = [r["macro avg"]["recall"] for r in res["fold_reports"]]
    
    cell_acc = fmt(res["acc_mean"], res["acc_std"])
    cell_mcc = fmt(res["mcc_mean"], res["mcc_std"])
    cell_prec = fmt(np.mean(f_precisions), np.std(f_precisions))
    cell_rec = fmt(np.mean(f_recalls), np.std(f_recalls))
    cell_f1 = fmt(res["f1_mean"], res["f1_std"])
    
    line = (f"{model_name:25s}"
            f"{cell_acc:>{col_w}}"
            f"{cell_mcc:>{col_w}}"
            f"{cell_prec:>{col_w}}"
            f"{cell_rec:>{col_w}}"
            f"{cell_f1:>{col_w}}")
    print(line)

print(f"{'='*100}\n")
