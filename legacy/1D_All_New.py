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

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# ==== AYARLAR ====
label_mapping = {
    'bronchiectasis': 0,
    'bronchiolitis': 1,
    'copd': 2,
    'healthy': 3,
    'lrti': 4,
    'pneumonia': 5,
    'urti': 6
}
class_labels = list(label_mapping.keys())
data_dir = "/Users/okanduzyel/Downloads/Last_ICHBI2"
N_FOLDS = 5
SAMPLE_RATE = 22050

# === Veri Artırma Fonksiyonları ===
def add_noise(y, noise_level=0.005):
    return y + noise_level * np.random.randn(len(y))

def pitch_shift(y, sr, steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

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
filepaths, labels, crackles, wheezes = [], [], [], []
for fold_name in sorted(os.listdir(data_dir)):
    fold_dir = os.path.join(data_dir, fold_name)
    if not os.path.isdir(fold_dir): continue
    for class_name in sorted(os.listdir(fold_dir)):
        if class_name.lower() == "asthma":
            continue  # asthma sınıfını dışarıda bırak
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

filepaths = np.array(filepaths)
labels = np.array(labels)
crackles = np.array(crackles)
wheezes = np.array(wheezes)

print(f"Toplam {len(filepaths)} ses dosyası bulundu.")

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
        num_class=len(class_labels), random_state=42, eval_metric='mlogloss'
    )
}

# ==== 5-Fold Cross-Validation ve Değerlendirme ====
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for model_name, clf in classifiers.items():
    print(f"\n{'='*50}\nEğitiliyor ve Test Ediliyor: {model_name}\n{'='*50}")
    
    all_true, all_pred = [], []
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # Kalan tüm modeller makaleye göre SMOTE ile beslenir
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        # Modeli eğit
        clf.fit(X_train_res, y_train_res)
        
        # Test et
        y_pred = clf.predict(X_test)
        
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    # Sonuçları Yazdır
    acc = accuracy_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)
    
    print(f"{model_name} - 5 Fold Genel Sonuçlar")
    print(f"Accuracy : {acc:.4f}")
    print(f"MCC      : {mcc:.4f}\n")
    print(classification_report(all_true, all_pred, target_names=class_labels, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_true, all_pred))
