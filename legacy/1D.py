import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import re

# ==== AYARLAR ====
label_mapping = {'bronchiectasis': 0, 'bronchiolitis': 1, 'copd': 2, 'healthy': 3, 'pneumonia': 4, 'urti': 5}
class_labels = list(label_mapping.keys())
data_dir = "/Users/okanduzyel/Downloads/Last_ICHBI"
N_FOLDS = 5
SAMPLE_RATE = 22050

def extract_features(y, sr, crackle, wheeze):
    feats = []
    # MFCC (20 boyut)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(np.mean(zcr))
    feats.append(np.std(zcr))
    # RMS
    rms = librosa.feature.rms(y=y)
    feats.append(np.mean(rms))
    feats.append(np.std(rms))
    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.append(np.mean(centroid))
    feats.append(np.std(centroid))
    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats.append(np.mean(bandwidth))
    feats.append(np.std(bandwidth))
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.append(np.mean(rolloff))
    feats.append(np.std(rolloff))
    # Crackle/Wheeze (ek feature)
    feats.append(crackle)
    feats.append(wheeze)
    return np.array(feats)

def parse_crackle_wheeze_from_filename(filename):
    match = re.search(r'_cr([01])_wh([01])', filename)
    if match:
        crackle = int(match.group(1))
        wheeze = int(match.group(2))
    else:
        crackle = 0
        wheeze = 0
    return crackle, wheeze

filepaths, labels, crackles, wheezes = [], [], [], []
for fold_name in sorted(os.listdir(data_dir)):
    fold_dir = os.path.join(data_dir, fold_name)
    if not os.path.isdir(fold_dir):
        continue
    for class_name in sorted(os.listdir(fold_dir)):
        class_dir = os.path.join(fold_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith('.wav'):
                filepaths.append(os.path.join(class_dir, filename))
                labels.append(label_mapping[class_name])
                crackle, wheeze = parse_crackle_wheeze_from_filename(filename)
                crackles.append(crackle)
                wheezes.append(wheeze)
filepaths = np.array(filepaths)
labels = np.array(labels)
crackles = np.array(crackles)
wheezes = np.array(wheezes)

print(f"Toplam {len(filepaths)} ses dosyası bulundu.")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Her model için fold bazlı kayıt tut
model_names = ["SVM", "RandomForest", "LogisticRegression", "KNN", "MLP"]
model_objs = {
    "SVM": SVC(probability=True, kernel="linear"),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

all_true_models = {mn: [] for mn in model_names}
all_pred_models = {mn: [] for mn in model_names}

for fold_num, (train_idx, test_idx) in enumerate(skf.split(filepaths, labels), 1):
    print(f"\n=== Fold {fold_num}/{N_FOLDS} ===")
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in train_idx:
        y, sr = librosa.load(filepaths[i], sr=SAMPLE_RATE)
        X_train.append(extract_features(y, sr, crackles[i], wheezes[i]))
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

    for model_name in model_names:
        model = model_objs[model_name]
        print(f"\n--- {model_name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=class_labels, digits=4))
        print("MCC:", matthews_corrcoef(y_test, y_pred))
        all_true_models[model_name].extend(y_test.tolist())
        all_pred_models[model_name].extend(y_pred.tolist())

# --- Her model için ayrı toplam rapor ---
for model_name in model_names:
    print(f"\n==== {model_name} - TOPLAM (5 Fold) ====")
    print(classification_report(all_true_models[model_name], all_pred_models[model_name], target_names=class_labels, digits=4))
    print("MCC:", matthews_corrcoef(all_true_models[model_name], all_pred_models[model_name]))
    print("Confusion Matrix:\n", confusion_matrix(all_true_models[model_name], all_pred_models[model_name]))
