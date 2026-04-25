# generate_predictions.py

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.special import softmax

# --- AYARLAR ---
MODEL_PATH = "./model/checkpoint-141220"
TEST_DATA_PATH = "test_data_cleaned.csv"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Kullanılan Cihaz: {DEVICE}")

# --- MODEL VE VERİYİ YÜKLE ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenemedi. Hata: {e}")
    exit()

# Veri setini yükle
df_test = pd.read_csv(TEST_DATA_PATH).dropna()
# Etiketleri 0'dan başlayan index'e çevir (1-5 -> 0-4)
df_test['labels'] = df_test['star_rating'] - 1 

# PyTorch Dataset
class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

test_dataset = TestDataset(
    texts=df_test['review_text'].tolist(),
    labels=df_test['labels'].tolist(),
    tokenizer=tokenizer
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- TAHMİNLERİ YAP ---
all_true_labels = []
all_pred_labels = []
all_pred_probas = []

print("Test seti üzerinde tahminler yapılıyor...")
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Tahminler"):
        labels = batch['labels'].numpy()
        all_true_labels.extend(labels)
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Olasılıkları hesapla (softmax)
        probabilities = softmax(logits.cpu().numpy(), axis=1)
        all_pred_probas.extend(probabilities)
        
        # Tahmin edilen sınıfları al (argmax)
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_pred_labels.extend(predictions)

# --- SONUÇLARI KAYDET ---
np.savez_compressed(
    'test_predictions.npz',
    y_true=np.array(all_true_labels),
    y_pred=np.array(all_pred_labels),
    y_proba=np.array(all_pred_probas)
)

print("\nTahminler başarıyla 'test_predictions.npz' dosyasına kaydedildi.")
print(f"Toplam {len(all_true_labels)} örnek işlendi.")