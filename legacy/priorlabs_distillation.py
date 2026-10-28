import os
import numpy as np
import pandas as pd
import warnings
import socket
import time
import pickle

from scipy import stats

# ==== TabPFN Cloud API İstemcisi Entegrasyonu ====
import tabpfn_client
from tabpfn_client import TabPFNClassifier, set_access_token

# ==== Öğrenci Modeller (Student Regressors) ====
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# ==== Metrikler ve Bölümleme ====
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

# ==== PyTorch (Distillation Altyapısı) ====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# macOS OpenMP thread çakışma yaması
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==== AUTHENTICATE (PRIOR LABS CLOUD API) ====
API_TOKEN = os.environ.get("PRIORLABS_TOKEN", "")  # set the key as an environment variable
tabpfn_client.set_access_token(API_TOKEN)

# ==== AYARLAR (CSV TABANLI) ====
INPUT_CSV = "respiratory_dataset.csv"
TARGET_COL = "label_id"
DROP_COLS = ["filename", "patient_id", "label", "split"]
N_FOLDS = 5
N_REPEATS = 3

CACHE_FILE = "distillation_checkpoint.pkl"

label_mapping = {
    'bronchiectasis': 0, 'bronchiolitis': 1, 'copd': 2,
    'healthy': 3, 'pneumonia': 4, 'urti': 5
}
class_labels = list(label_mapping.keys())
NUM_CLASSES = len(class_labels)

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

X_all = df.drop(columns=DROP_COLS + [TARGET_COL])
y_all = df[TARGET_COL].to_numpy()
patient_ids = df["patient_id"].to_numpy()

INPUT_DIM = X_all.shape[1]

print(f"Toplam {len(df)} satır yüklendi.")
print(f"Özellik boyutu (Feature Dimension): {INPUT_DIM}")
print(f"Benzersiz hasta sayısı: {len(np.unique(patient_ids))}")

# ==============================================================
# STUDENT MIMARILERI
# ==============================================================
class SmallTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

class SmallMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_student_pytorch(model, X_tr, y_tr, P_teacher, epochs=100, batch_size=64, lr=1e-3, alpha=0.9):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    X_tr_np = X_tr.to_numpy() if isinstance(X_tr, pd.DataFrame) else X_tr
    X_tr_np = X_tr_np.astype(np.float32)
    P_teacher = P_teacher.astype(np.float32)
    dataset = TensorDataset(torch.FloatTensor(X_tr_np), torch.LongTensor(y_tr), torch.FloatTensor(P_teacher))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch, p_teach_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss_hard = ce_loss(logits, y_batch)
            log_probs = nn.functional.log_softmax(logits, dim=1)
            loss_soft = kl_loss(log_probs, p_teach_batch)
            loss = alpha * loss_soft + (1 - alpha) * loss_hard
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model

def predict_pytorch(model, X_te):
    model.eval()
    X_te_np = X_te.to_numpy() if isinstance(X_te, pd.DataFrame) else X_te
    X_te_np = X_te_np.astype(np.float32)
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_te_np))
        probs = nn.functional.softmax(logits, dim=1)
    return torch.argmax(probs, dim=1).numpy()

# ==============================================================
# CHECKPOINT KONTROLÜ (RESUME MEKANİZMASI)
# ==============================================================
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        checkpoint = pickle.load(f)
    results = checkpoint["results"]
    saved_fold_outputs = checkpoint["saved_fold_outputs"]
    print(f"♻️ Bulunan checkpoint yüklendi. Daha önce {len(saved_fold_outputs)} adet fold tamamlanmış.")
else:
    results = {
        "Teacher_TabPFN": {"fold_accs": [], "fold_mccs": [], "fold_reports": []},
        "Student_XGBoost": {"fold_accs": [], "fold_mccs": [], "fold_reports": []},
        "Student_DecisionTree": {"fold_accs": [], "fold_mccs": [], "fold_reports": []},
        "Student_MLP": {"fold_accs": [], "fold_mccs": [], "fold_reports": []},
        "Student_Transformer": {"fold_accs": [], "fold_mccs": [], "fold_reports": []}
    }
    saved_fold_outputs = {}

def record_fold_metrics(model_key, y_true_fold, y_pred_fold):
    results[model_key]["fold_accs"].append(accuracy_score(y_true_fold, y_pred_fold))
    results[model_key]["fold_mccs"].append(matthews_corrcoef(y_true_fold, y_pred_fold))
    results[model_key]["fold_reports"].append(classification_report(
        y_true_fold, y_pred_fold, labels=list(label_mapping.values()),
        target_names=class_labels, digits=4, output_dict=True, zero_division=0
    ))

print(f"\n{'='*75}\n🚀 CLOUD API TABPFN 15-FOLD DISTILLATION DÖNGÜSÜ BAŞLIYOR\n{'='*75}")

global_fold_counter = 1
max_retries = 6

for repeat in range(N_REPEATS):
    current_seed = 42 + repeat
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=current_seed)

    for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups=patient_ids), 1):
        fold_key = f"r_{repeat}_f_{fold_num}"
        print(f"\n--- Global Fold {global_fold_counter}/15 (Repeat {repeat+1}, Fold {fold_num}/5) ---")
        
        X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
        X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
        X_train_np = X_train.to_numpy().astype(np.float32)
        X_test_np = X_test.to_numpy().astype(np.float32)

        # Önbellek (Cache) Kontrol Adımı
        if fold_key in saved_fold_outputs:
            print(f"📦 [CACHE] Bu fold verileri lokal diskten okundu, API çağrısı atlanıyor...")
            y_pred_teacher = saved_fold_outputs[fold_key]["y_pred_teacher"]
            P_teacher_train = saved_fold_outputs[fold_key]["P_teacher_train"]
        else:
            # 1. CLOUD TEACHER (Exponential Backoff ile Güçlendirildi)
            teacher_fitted = False
            base_delay = 15
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"☁️ Cloud API Teacher (TabPFN) çağrılıyor (Deneme {attempt}/{max_retries})...")
                    teacher = TabPFNClassifier(n_estimators=8)
                    teacher.fit(X_train, y_train)
                    y_pred_teacher = teacher.predict(X_test)
                    
                    print(f"🧪 Bulut üzerinden soft label olasılıkları indiriliyor...")
                    P_teacher_train = teacher.predict_proba(X_train)
                    teacher_fitted = True
                    break
                except Exception as e:
                    current_delay = base_delay * (2 ** (attempt - 1))
                    print(f"⚠️ Bağlantı hatası/Zaman aşımı: {e}")
                    if attempt < max_retries:
                        print(f"⏳ Ağ kilitlenmesi şüphesi. {current_delay} saniye bekleniyor...")
                        time.sleep(current_delay)
                    else:
                        raise e

            # Fold çıktılarını diske yedekle
            saved_fold_outputs[fold_key] = {
                "y_pred_teacher": y_pred_teacher,
                "P_teacher_train": P_teacher_train
            }

        record_fold_metrics("Teacher_TabPFN", y_test, y_pred_teacher)
        P_teacher_train_np = P_teacher_train.astype(np.float32)

        # 2. STUDENT 1: XGBOOST REGRESSOR
        print("🎓 Student 1 (XGBoost) damıtılıyor...")
        student_xgb = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=1)
        student_xgb.fit(X_train_np, P_teacher_train_np)
        xgb_probs = student_xgb.predict(X_test_np)
        record_fold_metrics("Student_XGBoost", y_test, np.argmax(xgb_probs, axis=1))

        # 3. STUDENT 2: DECISION TREE REGRESSOR
        print("🎓 Student 2 (Decision Tree) damıtılıyor...")
        student_dt = DecisionTreeRegressor(max_depth=None, min_samples_split=4, min_samples_leaf=2, random_state=42)
        student_dt.fit(X_train_np, P_teacher_train_np)
        dt_probs = student_dt.predict(X_test_np)
        record_fold_metrics("Student_DecisionTree", y_test, np.argmax(dt_probs, axis=1))

        # 4. STUDENT 3: SMALL MLP
        print("🎓 Student 3 (Small MLP) damıtılıyor...")
        student_mlp = SmallMLP(INPUT_DIM, NUM_CLASSES)
        student_mlp = train_student_pytorch(student_mlp, X_train_np, y_train, P_teacher_train_np, epochs=100, batch_size=64, lr=0.002, alpha=0.9)
        record_fold_metrics("Student_MLP", y_test, predict_pytorch(student_mlp, X_test_np))

        # 5. STUDENT 4: SMALL TRANSFORMER
        print("🎓 Student 4 (Small Transformer) damıtılıyor...")
        student_tf = SmallTransformer(INPUT_DIM, NUM_CLASSES, d_model=256, nhead=8, num_layers=4)
        student_tf = train_student_pytorch(student_tf, X_train_np, y_train, P_teacher_train_np, epochs=120, batch_size=64, lr=0.001, alpha=0.95)
        record_fold_metrics("Student_Transformer", y_test, predict_pytorch(student_tf, X_test_np))

        # Her fold bitiminde kontrol noktasını güncelle (Diske kaydet)
        checkpoint_data = {"results": results, "saved_fold_outputs": saved_fold_outputs}
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(checkpoint_data, f)

        global_fold_counter += 1

# ==============================================================================
# NİHAİ AKADEMİK RAPORLAMA ÇIKTILARI (15-FOLD ÖZET PERFORMANS TABLO)
# ==============================================================================
print(f"\n{'='*120}")
print(f"🏆 NİHAİ 15-FOLD DISTILLATION ÖZET PERFORMANS TABLOSU (PATIENT-LEVEL)")
print(f"{'='*120}")

col_name, col_m = 25, 18
header = f"{'Classifier':<{col_name}}{'Accuracy':<{col_m}}{'MCC':<{col_m}}{'Macro Precision':<{col_m}}{'Macro Recall':<{col_m}}{'Macro F1':<{col_m}}{'Weighted F1':<{col_m}}"
print(header)
print("-" * (col_name + col_m * 6))

def fmt_table(means_list, stds_list):
    return f"{np.mean(means_list):.4f} ± {np.std(stds_list):.4f}"

for model_key, res in results.items():
    f_precisions = [r["macro avg"]["precision"] for r in res["fold_reports"]]
    f_recalls = [r["macro avg"]["recall"] for r in res["fold_reports"]]
    f_macro_f1s = [r["macro avg"]["f1-score"] for r in res["fold_reports"]]
    f_weighted_f1s = [r["weighted avg"]["f1-score"] for r in res["fold_reports"]]
    
    display_name = model_key.replace("_", " ")
    if "TabPFN" in display_name and "Plus" not in display_name:
        display_name = display_name.replace("TabPFN", "TabPFN3-Plus")
    
    print(f"{display_name:<{col_name}}"
          f"{fmt_table(res['fold_accs'], res['fold_accs']):<{col_m}}"
          f"{fmt_table(res['fold_mccs'], res['fold_mccs']):<{col_m}}"
          f"{fmt_table(f_precisions, f_precisions):<{col_m}}"
          f"{fmt_table(f_recalls, f_recalls):<{col_m}}"
          f"{fmt_table(f_macro_f1s, f_macro_f1s):<{col_m}}"
          f"{fmt_table(f_weighted_f1s, f_weighted_f1s):<{col_m}}")

print(f"{'='*120}\n")

# ==============================================================================
# ==== 15-FOLD NADEAU-BENGIO DISTILLATION FIDELITY MATRIX ====
# ==============================================================================
print(f"\n\n{'='*110}")
print(f"🚨 NADEAU-BENGIO REPEATED FIDELITY MATRIX (15 FOLDS: Teacher TabPFN3 vs. Students)")
print(f"{'='*110}")

def nadeau_bengio_distillation_test(scores_teacher, scores_student, n_folds=5, n_repeats=3):
    diff = np.array(scores_teacher) - np.array(scores_student)
    n_total = n_folds * n_repeats
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)
    if var_diff == 0:
        return mean_diff, (np.inf if mean_diff != 0 else 0.0), (0.0 if mean_diff != 0 else 1.0)
    
    correction = (1 / n_total) + 0.25
    denom = np.sqrt(correction * var_diff)
    t_stat = mean_diff / denom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_total - 1))
    return mean_diff, t_stat, p_value

metrics_to_test = ["mcc", "accuracy", "macro_f1"]
teacher_key = "Teacher_TabPFN"
student_keys = [k for k in results.keys() if k != teacher_key]

for metric in metrics_to_test:
    print(f"\n--- Distillation Quality Analysis for Metric: {metric.upper()} (15 Folds) ---")
    print(f"{'Teacher Model (A)':22s}{'Student Model (B)':24s}{'Mean Loss (A-B)':>16s}{'t-Statistic':>14s}{'p-value':>12s}{'Fidelity Verdict':>22s}")
    print("-" * 110)
    
    if metric == "macro_f1":
        scores_teacher = [r["macro avg"]["f1-score"] for r in results[teacher_key]["fold_reports"]]
    else:
        scores_teacher = results[teacher_key]["fold_accs" if metric == "accuracy" else "fold_mccs"]
    scores_teacher = np.array(scores_teacher)
    
    for student_key in student_keys:
        if metric == "macro_f1":
            scores_student = [r["macro avg"]["f1-score"] for r in results[student_key]["fold_reports"]]
        else:
            scores_student = results[student_key]["fold_accs" if metric == "accuracy" else "fold_mccs"]
        scores_student = np.array(scores_student)
        
        mean_diff, t_stat, p_value = nadeau_bengio_distillation_test(scores_teacher, scores_student, n_folds=N_FOLDS, n_repeats=N_REPEATS)
        verdict = "OK (Successful Parity)" if p_value >= 0.05 else "DROP (Significant Loss)"
        p_str = f"{p_value:.5f}" if p_value >= 0.001 else "< 0.001"
        print(f"{'TabPFN3-Plus':22s}{student_key.replace('_', ' '):24s}{mean_diff:16.4f}{t_stat:14.4f}{p_str:>12s}{verdict:>22s}")

print(f"{'='*110}\n")

# Görev bitiminde önbellek dosyasını temizleyelim
if os.path.exists(CACHE_FILE):
    os.remove(CACHE_FILE)
