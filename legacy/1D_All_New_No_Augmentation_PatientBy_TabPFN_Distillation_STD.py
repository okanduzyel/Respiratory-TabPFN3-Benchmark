import os
import numpy as np
import librosa
import random
import re
import warnings

# ==== Kütüphaneler ====
from xgboost import XGBRegressor # Öğrenci için Regressor kullanılır
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor # Öğrenci DT için Regressor
from tabpfn import TabPFNClassifier

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

# ==== Pytorch (Distillation İçin) ====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==== AYARLAR ====
label_mapping = {
    'bronchiectasis': 0,
    'bronchiolitis': 1,
    'copd': 2,
    'healthy': 3,
    'pneumonia': 4,
    'urti': 5
}
class_labels = list(label_mapping.keys())
NUM_CLASSES = len(class_labels)
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
        if class_name.lower() in ["asthma", "lrti"]:
            continue
        class_dir = os.path.join(fold_dir, class_name)
        if not os.path.isdir(class_dir): continue
        if class_name.lower() not in label_mapping:
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
INPUT_DIM = X_all.shape[1]

# ==============================================================
# PYTORCH ÖĞRENCİ MODELLERİ (STUDENT MODELS)
# ==============================================================

# 1. Küçük Transformer Öğrencisi
class SmallTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # (batch, seq_len=1, features)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# 2. Küçük MLP Öğrencisi
class SmallMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# PyTorch Distillation Eğitim Döngüsü (KL Divergence Loss)
def train_student_pytorch(model, X_tr, y_tr, P_teacher, epochs=30, batch_size=32, lr=1e-3, alpha=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr), torch.FloatTensor(P_teacher))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch, p_teach_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            
            # Gerçek Etiket (Hard Loss)
            loss_hard = ce_loss(logits, y_batch)
            
            # Öğretmenden Gelen Soft Etiket (Soft Loss)
            log_probs = nn.functional.log_softmax(logits, dim=1)
            loss_soft = kl_loss(log_probs, p_teach_batch)
            
            # Toplam Loss: α * Soft_Loss + (1-α) * Hard_Loss
            loss = alpha * loss_soft + (1 - alpha) * loss_hard
            loss.backward()
            optimizer.step()
            
    return model

def predict_pytorch(model, X_te):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_te))
        probs = nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()
    return preds

# ==============================================================
# EĞİTİM & BİLGİ DAMITMA (DISTILLATION) DÖNGÜSÜ
# ==============================================================
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Sonuçları tutmak için sözlük
results = {
    "Teacher_TabPFN": {"true": [], "pred": [], "fold_accs": [], "fold_mccs": [], "fold_reports": []},
    "Student_XGBoost": {"true": [], "pred": [], "fold_accs": [], "fold_mccs": [], "fold_reports": []},
    "Student_DecisionTree": {"true": [], "pred": [], "fold_accs": [], "fold_mccs": [], "fold_reports": []},
    "Student_MLP": {"true": [], "pred": [], "fold_accs": [], "fold_mccs": [], "fold_reports": []},
    "Student_Transformer": {"true": [], "pred": [], "fold_accs": [], "fold_mccs": [], "fold_reports": []}
}

def record_fold_metrics(model_key, y_true_fold, y_pred_fold):
    results[model_key]["fold_accs"].append(accuracy_score(y_true_fold, y_pred_fold))
    results[model_key]["fold_mccs"].append(matthews_corrcoef(y_true_fold, y_pred_fold))
    results[model_key]["fold_reports"].append(classification_report(
        y_true_fold, y_pred_fold,
        labels=list(label_mapping.values()),
        target_names=class_labels,
        digits=4,
        output_dict=True,
        zero_division=0
    ))

print(f"\n{'='*60}\n🚀 KNOWLEDGE DISTILLATION (BİLGİ DAMITMA) BAŞLIYOR\n{'='*60}")

for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=patient_ids), 1):
    print(f"\n--- FOLD {fold_num} ---")
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # 1. TEACHER (TabPFN) EĞİTİMİ (Akıllı Alt Küme ile)
    if len(X_train) > TABPFN_MAX_SAMPLES:
        sample_indices = np.random.choice(len(X_train), TABPFN_MAX_SAMPLES, replace=False)
        X_train_tab = X_train[sample_indices]
        y_train_tab = y_train[sample_indices]
    else:
        X_train_tab, y_train_tab = X_train, y_train

    print(f"👨‍🏫 Teacher (TabPFN) eğitiliyor ({len(X_train_tab)} örnek ile)...")
    teacher = TabPFNClassifier(device='cpu')
    teacher.fit(X_train_tab, y_train_tab)
    
    y_pred_teacher = teacher.predict(X_test)
    results["Teacher_TabPFN"]["true"].extend(y_test)
    results["Teacher_TabPFN"]["pred"].extend(y_pred_teacher)
    record_fold_metrics("Teacher_TabPFN", y_test, y_pred_teacher)

    # 2. SOFT LABEL ÜRETİMİ (TÜM VERİ İÇİN!)
    print(f"🧪 Öğrenciler için Soft Label üretiliyor (Tüm {len(X_train)} eğitim örneğine)...")
    P_teacher_train = teacher.predict_proba(X_train)

    # 3. STUDENT 1: XGBOOST
    print("🎓 Student 1 (XGBoost) damıtılıyor...")
    student_xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42))
    student_xgb.fit(X_train, P_teacher_train)
    xgb_probs = student_xgb.predict(X_test)
    y_pred_xgb = np.argmax(xgb_probs, axis=1)
    results["Student_XGBoost"]["true"].extend(y_test)
    results["Student_XGBoost"]["pred"].extend(y_pred_xgb)
    record_fold_metrics("Student_XGBoost", y_test, y_pred_xgb)

    # 4. STUDENT 2: DECISION TREE (Açıklanabilir Yapay Zeka için Eklendi!)
    print("🎓 Student 2 (Decision Tree) damıtılıyor...")
    # Sınıflandırıcı değil, olasılıkları öğrenmesi için Regressor kullanıyoruz
    student_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    student_dt.fit(X_train, P_teacher_train)
    dt_probs = student_dt.predict(X_test)
    y_pred_dt = np.argmax(dt_probs, axis=1)
    results["Student_DecisionTree"]["true"].extend(y_test)
    results["Student_DecisionTree"]["pred"].extend(y_pred_dt)
    record_fold_metrics("Student_DecisionTree", y_test, y_pred_dt)

    # 5. STUDENT 3: KÜÇÜK MLP (PyTorch)
    print("🎓 Student 3 (Small MLP) damıtılıyor...")
    student_mlp = SmallMLP(INPUT_DIM, NUM_CLASSES)
    student_mlp = train_student_pytorch(student_mlp, X_train, y_train, P_teacher_train, epochs=40, lr=0.005)
    y_pred_mlp = predict_pytorch(student_mlp, X_test)
    results["Student_MLP"]["true"].extend(y_test)
    results["Student_MLP"]["pred"].extend(y_pred_mlp)
    record_fold_metrics("Student_MLP", y_test, y_pred_mlp)

    # 6. STUDENT 4: KÜÇÜK TRANSFORMER (PyTorch)
    print("🎓 Student 4 (Small Transformer) damıtılıyor...")
    student_tf = SmallTransformer(INPUT_DIM, NUM_CLASSES)
    student_tf = train_student_pytorch(student_tf, X_train, y_train, P_teacher_train, epochs=40, lr=0.002)
    y_pred_tf = predict_pytorch(student_tf, X_test)
    results["Student_Transformer"]["true"].extend(y_test)
    results["Student_Transformer"]["pred"].extend(y_pred_tf)
    record_fold_metrics("Student_Transformer", y_test, y_pred_tf)


# ==============================================================
# GENEL SONUÇLARI YAZDIRMA
# ==============================================================
print(f"\n{'='*50}\n🏆 NİHAİ 5-FOLD DISTILLATION SONUÇLARI\n{'='*50}")

for model_name, res in results.items():
    all_t = res["true"]
    all_p = res["pred"]
    acc = accuracy_score(all_t, all_p)
    mcc = matthews_corrcoef(all_t, all_p)
    
    print(f"\n>>> {model_name.upper()} <<<")
    print(f"Accuracy : {acc:.4f}")
    print(f"MCC      : {mcc:.4f}\n")
    print(classification_report(all_t, all_p, target_names=class_labels, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_t, all_p))

    # mean ± std çıktısı
    print_metrics_with_std(model_name, res["fold_accs"], res["fold_mccs"], res["fold_reports"])
