# evaluate_performance.py

import torch
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

# --- 1. AYARLAR ---
MODEL_PATH = "./model/checkpoint-141220"
TEST_DATA_PATH = "test_data_cleaned.csv"
NUM_SAMPLES_LATENCY = 500  # Gecikme testi için kullanılacak örnek sayısı
BATCH_SIZE_THROUGHPUT = 32 # Verim testi için yığın boyutu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Kullanılan Cihaz: {DEVICE}")
print("-" * 30)

# --- 2. MODEL VE VERİYİ YÜKLE ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()  # Modeli değerlendirme moduna al
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenemedi. Hata: {e}")
    exit()

df_test = pd.read_csv(TEST_DATA_PATH).dropna()
test_texts = df_test['review_text'].tolist()

# Basit bir dataset sınıfı
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# --- 3. PERFORMANS TESTLERİ ---

@torch.no_grad() # Gradyan hesaplamasını kapatarak hızlandır
def test_latency():
    """Tekil örnekler için ortalama gecikme süresini ölçer."""
    print(f"\n{NUM_SAMPLES_LATENCY} örnek üzerinde Gecikme (Latency) testi başlatılıyor...")
    
    # GPU'yu "ısındırmak" için bir ön çalıştırma yap
    tokenizer("Bu bir ısınma metnidir.", return_tensors="pt").to(DEVICE)
    
    latencies = []
    # Rastgele örnekler seç
    sample_texts = np.random.choice(test_texts, NUM_SAMPLES_LATENCY, replace=False)

    for text in tqdm(sample_texts, desc="Gecikme Testi"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
        
        start_time = time.perf_counter()
        _ = model(**inputs)
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # saniyeyi milisaniyeye çevir

    avg_latency = np.mean(latencies)
    print(f"Ortalama Gecikme Süresi: {avg_latency:.2f} ms")
    return avg_latency

@torch.no_grad()
def test_throughput():
    """Toplu işleme için saniyedeki yorum sayısını (QPS) ölçer."""
    print(f"\n{len(test_texts)} örnek üzerinde Verim (Throughput) testi başlatılıyor (Batch Size={BATCH_SIZE_THROUGHPUT})...")

    dataset = InferenceDataset(test_texts, tokenizer)
    # Tokenizer'ı DataLoader'a bir fonksiyon olarak veriyoruz
    def collate_fn(batch_texts):
        return tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_THROUGHPUT, collate_fn=collate_fn)

    # GPU'yu "ısındırmak" için bir ön çalıştırma yap
    warmup_batch = next(iter(dataloader))
    _ = model(**{key: val.to(DEVICE) for key, val in warmup_batch.items()})

    total_time = 0
    total_samples = 0
    
    start_global_time = time.perf_counter()
    for batch in tqdm(dataloader, desc="Verim Testi"):
        inputs = {key: val.to(DEVICE) for key, val in batch.items()}
        _ = model(**inputs)
    end_global_time = time.perf_counter()
    
    total_time = end_global_time - start_global_time
    total_samples = len(test_texts)
    qps = total_samples / total_time
    
    print(f"Toplam Süre: {total_time:.2f} saniye")
    print(f"Saniyede İşlenen Yorum (QPS): {qps:.2f}")
    return qps

# --- 4. TESTLERİ ÇALIŞTIR ---
if __name__ == "__main__":
    print("Performans ve kaynak kullanımı testi için bu betik çalışırken,")
    print("yeni bir terminal açıp 'watch -n 0.5 nvidia-smi' komutunu çalıştırarak GPU VRAM kullanımını izleyin.")
    input("Devam etmek için Enter'a basın...")
    
    test_latency()
    test_throughput()