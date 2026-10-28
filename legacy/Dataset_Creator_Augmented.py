import os
import numpy as np
import pandas as pd
import librosa
import re
import warnings
from scipy.signal import butter, lfilter

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
data_dir = "/Users/okanduzyel/Downloads/Last_ICHBI2"
SAMPLE_RATE = 22050
OUTPUT_CSV = "respiratory_dataset_augmentation2.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# ==============================================================
# GELİŞMİŞ BİYOAKUSTİK AUGMENTATION TEKNİKLERİ
# ==============================================================
def butter_bandpass_filter(data, lowcut=150, highcut=2000, sr=22050, order=4):
    """ Stetoskop ve göğüs duvarı akustik filtrelemesi """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def augment_time_stretch(y, rate_range=(0.88, 1.12)):
    rate = np.random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(y, rate=rate)

def augment_pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
    if n_steps == 0:
        n_steps = 1
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_add_noise(y, snr_db=22):
    sig_power = np.mean(y ** 2)
    if sig_power == 0:
        return y
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def augment_mixup(y1, y2, alpha=0.2):
    """ İki aynı sınıf sesini konveks kombinasyonla birleştirir """
    lam = np.random.beta(alpha, alpha)
    # Boyutları eşitle
    min_len = min(len(y1), len(y2))
    return lam * y1[:min_len] + (1 - lam) * y2[:min_len]

def apply_random_advanced_augmentation(y_primary, y_secondary, sr):
    """ Çok katmanlı gelişmiş biyoakustik türetme motoru """
    y_aug = y_primary.copy()
    
    # %40 İhtimalle Mixup
    if y_secondary is not None and np.random.rand() < 0.4:
        y_aug = augment_mixup(y_aug, y_secondary)
        
    # %50 İhtimalle Stetoskop Bandpass Filtresi
    if np.random.rand() < 0.5:
        low = np.random.choice([100, 150, 200])
        high = np.random.choice([1800, 2000, 2200])
        y_aug = butter_bandpass_filter(y_aug, lowcut=low, highcut=high, sr=sr)

    # %50 İhtimalle Pitch/Time Stretch
    if np.random.rand() < 0.5:
        y_aug = augment_pitch_shift(y_aug, sr)
    else:
        y_aug = augment_time_stretch(y_aug)

    # %30 İhtimalle Hafif Arka Plan Gürültüsü
    if np.random.rand() < 0.3:
        y_aug = augment_add_noise(y_aug)

    return y_aug

# ==============================================================
# ÖZELLİK ÇIKARIMI
# ==============================================================
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

def build_feature_names():
    names = []
    names += [f"mfcc_mean_{i+1}" for i in range(20)]
    names += [f"mfcc_std_{i+1}" for i in range(20)]
    names += ["zcr_mean", "zcr_std"]
    names += ["rms_mean", "rms_std"]
    names += ["spectral_centroid_mean", "spectral_centroid_std"]
    names += ["spectral_bandwidth_mean", "spectral_bandwidth_std"]
    names += ["spectral_rolloff_mean", "spectral_rolloff_std"]
    names += ["crackle", "wheeze"]
    return names

# ==============================================================
# VERİ YÜKLEME VE TÜRETME DÖNGÜSÜ
# ==============================================================
file_records = []

for fold_name in sorted(os.listdir(data_dir)):
    fold_dir = os.path.join(data_dir, fold_name)
    if not os.path.isdir(fold_dir):
        continue
    for class_name in sorted(os.listdir(fold_dir)):
        if class_name.lower() in ["asthma", "lrti"]:
            continue
        class_dir = os.path.join(fold_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.lower() not in label_mapping:
            continue
            
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith('.wav'):
                path = os.path.join(class_dir, filename)
                cr, wh = parse_crackle_wheeze_from_filename(filename)
                patient_id = filename.split('_')[0]
                
                file_records.append({
                    "path": path,
                    "filename": filename,
                    "label": class_name.lower(),
                    "crackle": cr,
                    "wheeze": wh,
                    "patient_id": patient_id
                })

df_meta = pd.DataFrame(file_records)
class_counts = df_meta["label"].value_counts()
majority_class = class_counts.idxmax()
max_samples = class_counts.max()

rows = []
feature_names = build_feature_names()

for label_name, group in df_meta.groupby("label"):
    current_count = len(group)
    print(f"\n⚙️ [{label_name.upper()}] İşleniyor... (Orijinal Örnek: {current_count})")
    
    # 1. Orijinal verileri ekle
    for idx, row_info in group.iterrows():
        y, sr = librosa.load(row_info["path"], sr=SAMPLE_RATE)
        feats = extract_features(y, sr, row_info["crackle"], row_info["wheeze"])
        
        row_dict = {
            "filename": row_info["filename"],
            "patient_id": row_info["patient_id"],
            "label": row_info["label"]
        }
        row_dict.update(dict(zip(feature_names, feats)))
        rows.append(row_dict)
        
    # 2. Azınlık sınıflarını Mixup & Advanced Augmentation ile dengele
    if label_name != majority_class:
        needed_aug = max_samples - current_count
        print(f"   🚀 Gelişmiş Biyoakustik Augmentation: {needed_aug} yeni sentetik örnek üretiliyor...")
        
        group_records = group.to_dict('records')
        
        for aug_i in range(needed_aug):
            base_sample = group_records[aug_i % current_count]
            second_sample = group_records[(aug_i + 1) % current_count]
            
            y_prim, sr = librosa.load(base_sample["path"], sr=SAMPLE_RATE)
            y_sec, _ = librosa.load(second_sample["path"], sr=SAMPLE_RATE)
            
            y_aug = apply_random_advanced_augmentation(y_prim, y_sec, sr)
            
            feats_aug = extract_features(y_aug, sr, base_sample["crackle"], base_sample["wheeze"])
            
            aug_dict = {
                "filename": f"adv_aug_{aug_i+1}_{base_sample['filename']}",
                "patient_id": f"aug_{base_sample['patient_id']}",
                "label": base_sample["label"]
            }
            aug_dict.update(dict(zip(feature_names, feats_aug)))
            rows.append(aug_dict)

df_final = pd.DataFrame(rows)
df_final["label_id"] = df_final["label"].map(label_mapping)

# Hasta Bazlı Split
unique_patients = sorted(df_final["patient_id"].unique())
n_test_patients = max(1, int(round(len(unique_patients) * TEST_SIZE)))
test_patients_set = set(unique_patients[:n_test_patients])

df_final["split"] = df_final["patient_id"].apply(lambda pid: "test" if pid in test_patients_set else "train")

# CSV Yazımı
split_order = pd.Categorical(df_final["split"], categories=["test", "train"], ordered=True)
df_final = df_final.assign(_split_order=split_order).sort_values(
    by=["_split_order", "patient_id", "filename"]
).drop(columns="_split_order").reset_index(drop=True)

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\n🔥 Gelişmiş Dengelenmiş CSV Oluşturuldu: {OUTPUT_CSV}")
