import os
import numpy as np
import librosa
import re
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import joblib
from tabpfn import TabPFNClassifier

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

filepaths = np.array(filepaths)
labels = np.array(labels)
crackles = np.array(crackles)
wheezes = np.array(wheezes)

print(f"Toplam {len(filepaths)} ses dosyası bulundu.")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
all_true, all_pred = [], []

for fold_num, (train_idx, test_idx) in enumerate(skf.split(filepaths, labels), 1):
    print(f"\n=== Fold {fold_num}/{N_FOLDS} ===")
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in train_idx:
        y, sr = librosa.load(filepaths[i], sr=SAMPLE_RATE)
        cr, wh = crackles[i], wheezes[i]
        X_train.append(extract_features(y, sr, cr, wh))
        y_train.append(labels[i])
    for i in test_idx:
        y, sr = librosa.load(filepaths[i], sr=SAMPLE_RATE)
        X_test.append(extract_features(y, sr, crackles[i], wheezes[i]))
        y_test.append(labels[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=class_labels, digits=4))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    all_true.extend(y_test.tolist())
    all_pred.extend(y_pred.tolist())

print(f"\n==== TabPFN - TOPLAM (5 Fold, Sadece Orijinal) ====")
print(classification_report(all_true, all_pred, target_names=class_labels, digits=4))
print("MCC:", matthews_corrcoef(all_true, all_pred))
print("Confusion Matrix:\n", confusion_matrix(all_true, all_pred))

#---------------------------------------------------------------------

print("\n==== TÜM VERİ İLE FİNAL MODEL EĞİTİLİYOR ====")

X_all, y_all = [], []
for i, path in enumerate(filepaths):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    cr, wh = crackles[i], wheezes[i]
    X_all.append(extract_features(y, sr, cr, wh))
    y_all.append(labels[i])
X_all = np.array(X_all)
y_all = np.array(y_all)

final_model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
final_model.fit(X_all, y_all)

joblib.dump(final_model, "tabpfn_model_final.joblib")
print("Final TabPFN model tüm veriyle eğitildi ve 'tabpfn_model_final.joblib' olarak kaydedildi.")
