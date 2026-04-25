import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv('hb.csv')
df['processed_review'] = df['Review'].str.lower().str.strip()

X = df['processed_review'].tolist()
y = df['Rating (Star)'].values

label_map = {rating: i for i, rating in enumerate(sorted(set(y)))}
y_mapped = [label_map[rating] for rating in y]

X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=seed)

tokenizer = DistilBertTokenizer.from_pretrained('dbmdz/distilbert-base-turkish-cased')

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

train_dataset = ReviewDataset(X_train, y_train, tokenizer)
test_dataset = ReviewDataset(X_test, y_test, tokenizer)

model = DistilBertForSequenceClassification.from_pretrained(
    'dbmdz/distilbert-base-turkish-cased',
    num_labels=len(label_map)
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,                
    weight_decay=0.05,                 
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=seed,
    report_to="none",
    fp16=True,
    gradient_accumulation_steps=2
)

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)

predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

accuracy = accuracy_score(y_test, predicted_labels)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predicted_labels, target_names=[f"Star {i}" for i in label_map.keys()]))

model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_model')

