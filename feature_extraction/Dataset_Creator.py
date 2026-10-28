import os
import numpy as np
import pandas as pd
import librosa
import re
import warnings

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
OUTPUT_CSV = "respiratory_dataset.csv"
TEST_SIZE = 0.2       # hasta bazında ayrılacak test oranı
RANDOM_STATE = 42

# === Özellik Çıkarımı (orijinal koddaki ile birebir aynı) ===
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

# === Özellik isimleri (CSV kolon başlıkları için) ===
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

# === Veri Yükleme (dosya listesi) ===
filepaths, labels, crackles, wheezes, patient_ids = [], [], [], [], []

for fold_name in sorted(os.listdir(data_dir)):
    fold_dir = os.path.join(data_dir, fold_name)
    if not os.path.isdir(fold_dir):
        continue
    for class_name in sorted(os.listdir(fold_dir)):
        if class_name.lower() in ["asthma", "lrti"]:
            continue
        class_dir = os.path.join(fold_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name.lower() not in label_mapping:
            print(f"Uyarı: {class_name} label_mapping'te yok, atlanıyor.")
            continue
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith('.wav'):
                path = os.path.join(class_dir, filename)
                filepaths.append(path)
                labels.append(class_name.lower())  # CSV'de okunabilir olsun diye string tutuyoruz

                cr, wh = parse_crackle_wheeze_from_filename(filename)
                crackles.append(cr)
                wheezes.append(wh)

                patient_id = filename.split('_')[0]
                patient_ids.append(patient_id)

print(f"Toplam {len(filepaths)} ses dosyası bulundu.")
print(f"Benzersiz hasta sayısı: {len(np.unique(patient_ids))}")

# ==== Özellik çıkarımı ====
rows = []
feature_names = build_feature_names()

for i, path in enumerate(filepaths):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    cr, wh = crackles[i], wheezes[i]
    feats = extract_features(y, sr, cr, wh)

    row = {
        "filename": os.path.basename(path),
        "patient_id": patient_ids[i],
        "label": labels[i],
    }
    row.update(dict(zip(feature_names, feats)))
    rows.append(row)

    if (i + 1) % 50 == 0 or (i + 1) == len(filepaths):
        print(f"  İşlendi: {i + 1}/{len(filepaths)}")

df = pd.DataFrame(rows)

# İstenirse label_mapping numerik kodunu da ayrı kolon olarak ekleyelim (opsiyonel, TabPFN string label da kabul eder)
df["label_id"] = df["label"].map(label_mapping)

# ==== HASTA BAZLI, POZİSYONEL TRAIN/TEST SPLIT (CSV'nin ilk %20'si test, kalan %80'i train) ====
# Aynı hastaya ait tüm cycle'lar (kayıtlar) ayrık şekilde ya train ya da test'e düşer,
# böylece aynı hastanın farklı kayıtları train ve test arasında sızıntıya (leakage) yol açmaz.
unique_patients = sorted(df["patient_id"].unique())
n_test_patients = max(1, int(round(len(unique_patients) * TEST_SIZE)))
test_patients_set = set(unique_patients[:n_test_patients])

df["split"] = df["patient_id"].apply(lambda pid: "test" if pid in test_patients_set else "train")

# Sızıntı kontrolü: train/test hasta kümeleri kesinlikle ayrık olmalı
train_patients = set(df.loc[df["split"] == "train", "patient_id"])
test_patients = set(df.loc[df["split"] == "test", "patient_id"])
overlap = train_patients & test_patients
assert len(overlap) == 0, f"HATA: {len(overlap)} hasta hem train hem test'te! Split hatalı."

print(f"\nHasta bazlı split tamam: {len(train_patients)} train hasta, {len(test_patients)} test hasta, çakışma yok.")

# ==== CSV'yi test bloğu ÖNCE, train bloğu SONRA olacak şekilde, her blokta patient_id sırasına göre yaz ====
split_order = pd.Categorical(df["split"], categories=["test", "train"], ordered=True)
df = df.assign(_split_order=split_order).sort_values(
    by=["_split_order", "patient_id", "filename"]
).drop(columns="_split_order").reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV dosyası oluşturuldu: {OUTPUT_CSV}")
print(f"Boyut: {df.shape[0]} satır x {df.shape[1]} kolon")
print(df["split"].value_counts())
print(f"İlk satır sayısı (test): {(df['split'] == 'test').sum()}")
print(f"Son satır sayısı (train): {(df['split'] == 'train').sum()}")
