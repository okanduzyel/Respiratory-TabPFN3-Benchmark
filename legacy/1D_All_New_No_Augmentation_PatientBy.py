import os
import numpy as np
import librosa
import random
import re
import joblib

# ==== Kütüphaneler ====
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier

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
TABPFN_MAX_SAMPLES = 1000

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

# ==== MAKALEDEKİ OPTİMAL PARAMETRELERLE MODELLERİ TANIMLAMA ====
classifiers = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=42
    ),
    "KNN": KNeighborsClassifier(
        metric="manhattan", n_neighbors=3, weights="distance"
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=0.0001, learning_rate_init=0.001, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=1,
        criterion="entropy", bootstrap=False, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale", random_state=42
    ),
    "Naïve Bayes": GaussianNB(
        var_smoothing=1e-9
    ),
    "Logistic Regression": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        colsample_bytree=0.8, gamma=0, learning_rate=0.2, max_depth=4,
        n_estimators=200, subsample=0.8, objective='multi:softprob',
        random_state=42, eval_metric='mlogloss', use_label_encoder=False
    ),
    "TabPFN": TabPFNClassifier(device='cpu')
}

# ==== 5-Fold Cross-Validation (HASTA BAZLI / PATIENT-LEVEL) ====
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for model_name, clf in classifiers.items():
    print(f"\n{'='*50}\nEğitiliyor ve Test Ediliyor: {model_name}\n{'='*50}")
    
    all_true, all_pred = [], []
    
    for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=patient_ids), 1):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # TabPFN OOM Koruması
        if model_name == "TabPFN" and len(X_train) > TABPFN_MAX_SAMPLES:
            sample_indices = np.random.choice(len(X_train), TABPFN_MAX_SAMPLES, replace=False)
            X_train_fit = X_train[sample_indices]
            y_train_fit = y_train[sample_indices]
            print(f"  [!] Fold {fold_num}: TabPFN için {len(X_train)} örnekten {TABPFN_MAX_SAMPLES} kadarı rastgele seçildi.")
        else:
            X_train_fit = X_train
            y_train_fit = y_train

        # XGBoost Eksik Sınıf Koruması
        if model_name == "XGBoost":
            unique_train_classes = np.unique(y_train_fit)
            missing_classes = set(range(len(class_labels))) - set(unique_train_classes)
            
            if missing_classes:
                print(f"  [!] Fold {fold_num}: XGBoost için eksik sınıf tespit edildi: {missing_classes}. Dummy koruma eklendi.")
                for mc in missing_classes:
                    mc_index = np.where(y_all == mc)[0][0]
                    X_train_fit = np.vstack([X_train_fit, X_all[mc_index]])
                    y_train_fit = np.append(y_train_fit, mc)

        # Modeli eğit
        clf.fit(X_train_fit, y_train_fit)
        
        # Test et
        y_pred = clf.predict(X_test)
        
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
