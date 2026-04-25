import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('hb.csv')
df['processed_review'] = df['Review'].str.lower().str.strip()

X = df['processed_review'].tolist()
y = df['Rating (Star)'].values

label_map = {rating: i for i, rating in enumerate(sorted(set(y)))}
reverse_label_map = {v: k for k, v in label_map.items()}
y_mapped = [label_map[rating] for rating in y]

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)


tokenizer = DistilBertTokenizer.from_pretrained('./tezV3ELEKTRA/final_model')
model = DistilBertForSequenceClassification.from_pretrained('./tezV3ELEKTRA/final_model')

model.to(device)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

test_dataset = ReviewDataset(X_test, y_test, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32)

model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
target_names = [f"Star {reverse_label_map[i]}" for i in range(len(label_map))]
print(classification_report(all_labels, all_predictions, target_names=target_names))
