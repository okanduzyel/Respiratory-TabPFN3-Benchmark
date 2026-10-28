import os
import re
import warnings
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, lfilter

import tabpfn_client
from tabpfn_client import TabPFNClassifier, set_access_token
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# macOS OpenMP / Thread çakışmasını engellemek için çevre değişkenleri kilitleniyor
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==== Authenticate ====
API_TOKEN = os.environ.get("PRIORLABS_TOKEN", "")  # set the key as an environment variable
tabpfn_client.set_access_token(API_TOKEN)

# ==== AYARLAR ====
data_dir = "/Users/okanduzyel/Downloads/Last_ICHBI2"
SAMPLE_RATE = 22050
N_FOLDS = 5

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


# ==============================================================================
# 1. BİYOAKUSTİK FİLTRE VE AUGMENTATION MOTORU (WAV DÜZEYİ)
# ==============================================================================
def butter_bandpass_filter(data, lowcut=150, highcut=2000, sr=22050, order=4):
    """ Stetoskop ve göğüs duvarı akustik sönümleme filtresi """
    if len(data) < 32:  # Çok kısa sinyal koruması
        return data
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def augment_time_stretch(y, rate_range=(0.9, 1.1)):
    rate = np.random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(y, rate=rate)

def augment_pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
    if n_steps == 0: n_steps = 1
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_mixup(y1, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    min_len = min(len(y1), len(y2))
    return lam * y1[:min_len] + (1 - lam) * y2[:min_len]

def apply_audio_augmentation(y_prim, y_sec, sr):
    y_aug = y_prim.copy()
    if y_sec is not None and np.random.rand() < 0.4:
        y_aug = augment_mixup(y_aug, y_sec)
    if np.random.rand() < 0.5:
        y_aug = butter_bandpass_filter(y_aug, lowcut=150, highcut=2000, sr=sr)
    if np.random.rand() < 0.5:
        y_aug = augment_pitch_shift(y_aug, sr)
    else:
        y_aug = augment_time_stretch(y_aug)
    return y_aug


# ==============================================================================
# 2. GELİŞMİŞ ÖZELLİK ÇIKARIMI (ESNEK DELTA VE KISA SİNYAL KORUMALI)
# ==============================================================================
def extract_advanced_features(y, sr, crackle, wheeze):
    feats = []
    
    # Çok kısa sesler için Padding
    min_samples = int(sr * 0.5)  # En az 0.5 saniye
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)), mode='constant')
        
    # A. 20 MFCC mean + std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    
    # B. Delta MFCC mean + std (GÜVENLİ DELTA PENCERESİ: width=3)
    width = 3 if mfcc.shape[1] >= 3 else (mfcc.shape[1] if mfcc.shape[1] % 2 != 0 else mfcc.shape[1] - 1)
    if width >= 3:
        delta_mfcc = librosa.feature.delta(mfcc, width=width)
        feats.extend(np.mean(delta_mfcc, axis=1))
        feats.extend(np.std(delta_mfcc, axis=1))
    else:
        feats.extend(np.zeros(20))
        feats.extend(np.zeros(20))
    
    # C. Spectral Contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.extend(np.mean(contrast, axis=1))
        feats.extend(np.std(contrast, axis=1))
    except Exception:
        feats.extend(np.zeros(14))
    
    # D. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    feats.append(np.mean(flatness))
    feats.append(np.std(flatness))
    
    # E. Temel Frekans / Zaman Özellikleri
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
    
    # F. Patolojik Klinik Göstergeler
    feats.append(crackle)
    feats.append(wheeze)
    
    return np.array(feats)

def parse_crackle_wheeze_from_filename(filename):
    match = re.search(r'_cr([01])_wh([01])', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


# ==============================================================================
# 3. VERİ KÜMESİNİ DOĞRUDAN DİZİNDEN YÜKLEME
# ==============================================================================
print("📂 WAV Dosyaları Taranıyor...")
file_records = []

for fold_name in sorted(os.listdir(data_dir)):
    fold_dir = os.path.join(data_dir, fold_name)
    if not os.path.isdir(fold_dir): continue
    for class_name in sorted(os.listdir(fold_dir)):
        if class_name.lower() in ["asthma", "lrti"]: continue
        class_dir = os.path.join(fold_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.lower() not in label_mapping: continue
            
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith('.wav'):
                path = os.path.join(class_dir, filename)
                cr, wh = parse_crackle_wheeze_from_filename(filename)
                patient_id = filename.split('_')[0]
                
                file_records.append({
                    "path": path,
                    "filename": filename,
                    "label_str": class_name.lower(),
                    "label_id": label_mapping[class_name.lower()],
                    "crackle": cr,
                    "wheeze": wh,
                    "patient_id": patient_id
                })

df_meta = pd.DataFrame(file_records)
print(f"Toplam {len(df_meta)} ham WAV dosyası bulundu.")
print(f"Benzersiz hasta sayısı: {len(df_meta['patient_id'].unique())}")


# ==============================================================================
# 4. ON-THE-FLY FEATURE EXTRACTOR & ONLINE AUGMENTER
# ==============================================================================
def get_features_for_indices(indices_df, is_train_aug=False, random_state=42):
    np.random.seed(random_state)
    X_list, y_list = [], []
    
    if not is_train_aug:
        # TEST VEYA PLAIN TRAIN: Dokunulmadan orijinal sesler okunur
        for idx, row in indices_df.iterrows():
            y, sr = librosa.load(row["path"], sr=SAMPLE_RATE)
            feats = extract_advanced_features(y, sr, row["crackle"], row["wheeze"])
            X_list.append(feats)
            y_list.append(row["label_id"])
    else:
        # Sadece Train Kümesi İçin Dengelenmiş Online Augmentation
        class_counts = indices_df["label_id"].value_counts()
        majority_count = class_counts.max()
        
        for label_id, group in indices_df.groupby("label_id"):
            group_records = group.to_dict('records')
            curr_count = len(group_records)
            
            # Orijinal sesleri ekle
            for rec in group_records:
                y, sr = librosa.load(rec["path"], sr=SAMPLE_RATE)
                feats = extract_advanced_features(y, sr, rec["crackle"], rec["wheeze"])
                X_list.append(feats)
                y_list.append(rec["label_id"])
                
            # Azınlık sınıflarına online ses türetimi yap
            if curr_count < majority_count:
                needed = majority_count - curr_count
                for i in range(needed):
                    rec1 = group_records[i % curr_count]
                    rec2 = group_records[(i + 1) % curr_count]
                    
                    y1, sr = librosa.load(rec1["path"], sr=SAMPLE_RATE)
                    y2, _ = librosa.load(rec2["path"], sr=SAMPLE_RATE)
                    
                    y_aug = apply_audio_augmentation(y1, y2, sr)
                    feats_aug = extract_advanced_features(y_aug, sr, rec1["crackle"], rec1["wheeze"])
                    
                    X_list.append(feats_aug)
                    y_list.append(rec1["label_id"])
                    
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    
    if is_train_aug:
        shuf = np.random.permutation(len(y_arr))
        return pd.DataFrame(X_arr[shuf]), y_arr[shuf]
        
    return pd.DataFrame(X_arr), y_arr


# ==============================================================================
# 5. DENEY MOTORU (5-FOLD / ONLY TABPFN3)
# ==============================================================================
def run_experiment(strategy_id, strategy_name):
    print(f"\n\n{'='*90}")
    print(f"🧪 YÖNTEM {strategy_id} DENENİYOR: {strategy_name}")
    print(f"{'='*90}")
    
    accs, mccs, f1s, reports = [], [], [], []
    seed = 42
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(df_meta, df_meta["label_id"], groups=df_meta["patient_id"]), 1):
        train_df = df_meta.iloc[train_idx]
        test_df = df_meta.iloc[test_idx]  # TEST KÜMESİ SIZINTISIZ (%100 BAKİR)
        
        # --- TEST ÖZELLİKLERİ HER ZAMAN ORİJİNAL ÇIKARILIR ---
        X_te, y_te = get_features_for_indices(test_df, is_train_aug=False)
        
        # --- YÖNTEM BAZLI TRAIN ÖZELLİKLERİ ---
        is_aug = (strategy_id in [3, 8])
        X_tr, y_tr = get_features_for_indices(train_df, is_train_aug=is_aug, random_state=seed + fold_num)
        
        # Preprocessing (Quantile)
        if strategy_id == 7:
            qt = QuantileTransformer(output_distribution='normal', random_state=seed)
            X_tr = pd.DataFrame(qt.fit_transform(X_tr))
            X_te = pd.DataFrame(qt.transform(X_te))

        # Model Büyüklüğü (n_estimators)
        n_est = 32 if strategy_id in [4, 7, 8] else 8
        
        # In-Context Subsampling
        if strategy_id in [5, 8]:
            df_tr_all = X_tr.copy()
            df_tr_all['target'] = y_tr
            sub_dfs = []
            for cid, grp in df_tr_all.groupby('target'):
                if len(grp) > 350:
                    sub_dfs.append(grp.sample(n=350, random_state=seed))
                else:
                    sub_dfs.append(grp)
            df_sub = pd.concat(sub_dfs, axis=0).sample(frac=1.0, random_state=seed)
            X_tr = df_sub.drop(columns=['target'])
            y_tr = df_sub['target'].values

        # Model Eğitimi
        model = TabPFNClassifier(n_estimators=n_est)
        model.fit(X_tr, y_tr)
        
        # Tahmin & Post-processing
        if strategy_id in [2, 8]:
            classes = np.array(list(label_mapping.values()))
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df["label_id"].values)
            y_probs = model.predict_proba(X_te)
            w_probs = y_probs * weights
            w_probs = w_probs / w_probs.sum(axis=1, keepdims=True)
            y_pred = np.argmax(w_probs, axis=1)
            
        elif strategy_id == 6:
            y_probs = model.predict_proba(X_te)
            threshold_weights = np.array([1.6, 1.8, 0.7, 1.3, 1.5, 1.4])
            w_probs = y_probs * threshold_weights
            y_pred = np.argmax(w_probs, axis=1)
            
        else:
            y_pred = model.predict(X_te)
            
        accs.append(accuracy_score(y_te, y_pred))
        mccs.append(matthews_corrcoef(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro"))
        
        rep = classification_report(
            y_te, y_pred, labels=list(label_mapping.values()),
            target_names=class_labels, digits=4, output_dict=True, zero_division=0
        )
        reports.append(rep)
        print(f"   └─ Fold {fold_num}/5 Tamamlandı -> Fold MCC: {mccs[-1]:.4f}")
        
    print_results(strategy_name, accs, mccs, f1s, reports)
    return np.mean(mccs)


def print_results(strategy_name, accs, mccs, f1s, reports):
    print(f"\n{'='*85}")
    print(f"📊 {strategy_name} - 5 FOLD SONUÇ ÖZETİ")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"MCC      : {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    print(f"Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"{'='*85}")
    
    col_w = 17
    row_keys = class_labels + ["macro avg", "weighted avg"]
    metric_keys = ["precision", "recall", "f1-score"]
    
    collected = {row: {m: [] for m in metric_keys} for row in row_keys}
    fold_supports = {row: [] for row in class_labels}

    for report in reports:
        for row in row_keys:
            if row not in report: continue
            for m in metric_keys:
                collected[row][m].append(report[row][m])
        for label in class_labels:
            if label in report:
                fold_supports[label].append(report[label]["support"])

    header = f"{'Label / Metric':22s}{'precision':>{col_w}}{'recall':>{col_w}}{'f1-score':>{col_w}}{'support':>{col_w}}"
    print(header)
    print("-" * (22 + col_w * 4))

    for row in row_keys:
        if row not in collected or not collected[row]["precision"]: continue
        line = f"{row:22s}"
        for m in metric_keys:
            v = collected[row][m]
            cell = f"{np.mean(v):.4f}±{np.std(v):.4f}"
            line += f"{cell:>{col_w}}"

        if row in fold_supports:
            sup = int(np.mean(fold_supports[row]))
            line += f"{sup:>{col_w}}"
        else:
            total_sup = int(sum(np.mean(fold_supports[lbl]) for lbl in class_labels))
            line += f"{total_sup:>{col_w}}"
        print(line)


# ==============================================================================
# DENEY DÖNGÜSÜ BAŞLATILIYOR
# ==============================================================================
strategies = [
    (1, "Baseline (80+ Gelişmiş Özellik Havuzu - Ham Veri)"),
    (2, "Bayesian Cost-Sensitive Posterior Calibration"),
    (3, "Online Audio Augmentation (WAV Mixup & Stetoskop Filtresi)"),
    (4, "TabPFN3 Deep Ensemble (n_estimators=32)"),
    (5, "In-Context Subsampling (COPD Baskısını Dengeleme)"),
    (6, "Dynamic Threshold Tuning (Sınıf Eşik Ayarlaması)"),
    (7, "Quantile Normalization + Deep Ensemble (n_estimators=32)"),
    (8, "HİBRİT ZİRVE: Deep Ensemble + Cost-Sensitive + Online Audio Augmentation")
]

scores_summary = {}

for strategy_id, strategy_name in strategies:
    mcc_score = run_experiment(strategy_id, strategy_name)
    scores_summary[strategy_name] = mcc_score

print(f"\n\n{'='*90}")
print("🏆 TÜM YÖNTEMLERİN 5-FOLD MCC SKORLARI KARŞILAŞTIRMASI")
print(f"{'='*90}")
for name, score in sorted(scores_summary.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:55s} -> MCC: {score:.4f}")
