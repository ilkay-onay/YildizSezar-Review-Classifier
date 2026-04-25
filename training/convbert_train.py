import pandas as pd
import torch
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
import json
import shutil


# --- 1. Configuration and Setup ---
# Define a central config dictionary
CONFIG = {
    'seed': 42,
    'model_name': "dbmdz/convbert-base-turkish-mc4-uncased",
    'max_length': 256, # Added to config to make it a parameter
    'batch_size': 32,
    'eval_batch_size': 32,
    'epochs': 8,
    'learning_rate': 3e-5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 2,
    'early_stopping_patience': 3,
    'logging_steps': 10,
    'output_dir': "./results",  # Base output directory
    'log_dir': "./logs",   # Base log directory,
    "class_weight_mode": "balanced",  # Option: balanced, focal
    'fp16': torch.cuda.is_available()
}

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(train_file, val_file, test_file):
    # Load data
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    # Drop missing values
    df_train = df_train.dropna(subset=['review_text', 'star_rating'])
    df_val = df_val.dropna(subset=['review_text', 'star_rating'])
    df_test = df_test.dropna(subset=['review_text', 'star_rating'])

    # Convert star ratings to integers
    df_train['star_rating'] = df_train['star_rating'].astype(int)
    df_val['star_rating'] = df_val['star_rating'].astype(int)
    df_test['star_rating'] = df_test['star_rating'].astype(int)

    # Map star ratings to zero-based indices and keep a reversed mapping
    unique_labels = sorted(df_train['star_rating'].unique())
    label_map = {rating: i for i, rating in enumerate(unique_labels)}
    reverse_label_map = {i: rating for rating, i in label_map.items()}
    df_train['labels'] = df_train['star_rating'].map(label_map)
    df_val['labels'] = df_val['star_rating'].map(label_map)
    df_test['labels'] = df_test['star_rating'].map(label_map)

    return df_train, df_val, df_test, label_map, reverse_label_map

df_train, df_val, df_test, label_map, reverse_label_map = load_and_preprocess_data(
    'train_data_cleaned.csv', 'val_data_cleaned.csv', 'test_data_cleaned.csv'
)

# --- 3. Class Weight Calculation ---
def calculate_class_weights(labels, mode='balanced'):
    if mode == 'balanced':
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float)
    elif mode == 'focal':
        # TODO: Implement focal loss computation
        raise NotImplementedError("Focal Loss not implemented.")
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

class_weights = calculate_class_weights(df_train['labels'], mode=CONFIG['class_weight_mode'])

# --- 4. Dataset Preparation ---
# Reusable function for dataset creation
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
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
            padding="max_length", # Using max_length padding for more stability in distributed training
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
# Initialize tokenizer only once at the start
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

train_dataset = ReviewDataset(df_train['review_text'].tolist(), df_train['labels'].tolist(), tokenizer, CONFIG['max_length'])
val_dataset = ReviewDataset(df_val['review_text'].tolist(), df_val['labels'].tolist(), tokenizer, CONFIG['max_length'])
test_dataset = ReviewDataset(df_test['review_text'].tolist(), df_test['labels'].tolist(), tokenizer, CONFIG['max_length'])


# --- 5. Model Loading and Loss Function ---
# Reusable loss computation
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False): # Added num_items_in_batch argument
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=len(label_map)
)
# --- 6. Training Setup and Arguments ---

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(CONFIG['output_dir'], current_time)
log_dir = os.path.join(CONFIG['log_dir'], current_time)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=CONFIG['epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['eval_batch_size'],
    warmup_ratio=CONFIG['warmup_ratio'],
    weight_decay=CONFIG['weight_decay'],
    learning_rate=CONFIG['learning_rate'],
    lr_scheduler_type="cosine",
    max_grad_norm=1.0, # added as explicit arg, good to be explicit
    logging_dir=log_dir,
    logging_steps=CONFIG['logging_steps'],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=CONFIG['seed'],
    report_to="tensorboard",
    fp16=CONFIG['fp16'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
)


# --- 7. Metrics Calculation ---
def compute_metrics(pred, reverse_label_map, writer, epoch):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)

    # Log all metrics
    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # Confusion matrix to log metrics
    log_confusion_matrix(labels, predictions, list(reverse_label_map.values()), writer, epoch)
    return metrics

def log_confusion_matrix(y_true, y_pred, labels, writer, epoch):

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    writer.add_figure(f"Confusion_Matrix/Epoch_{epoch}", fig)

# --- 8. Model Training and Evaluation ---
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda pred: compute_metrics(pred, reverse_label_map, trainer.state.log_writer, trainer.state.epoch),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])]
)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)


print("Starting training...")
trainer.train()

print("Evaluating model...")
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)

# --- 9. Save Model and Test Results ---
model_dir = f'./final_model_{current_time}'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save configuration, tokenizer and label mapping.
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=4)
with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
    json.dump(reverse_label_map, f, indent=4)
print(f"Model, tokenizer, config and label mappings saved at {model_dir}!")

# Test Set Evaluation
test_predictions = trainer.predict(test_dataset)
test_predicted_labels = np.argmax(test_predictions.predictions, axis=1)

# Log the test confusion matrix and also save it to the log directory.
log_confusion_matrix(df_test['labels'], test_predicted_labels, list(reverse_label_map.values()), writer, "test")

# Test Accuracy
test_accuracy = accuracy_score(df_test['labels'], test_predicted_labels)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
writer.add_scalar("Test/Accuracy", test_accuracy, global_step=trainer.state.global_step)

# Classification Report
print("\nClassification Report:")
report = classification_report(df_test['labels'], test_predicted_labels, target_names=list(reverse_label_map.values()))
print(report)

# Save the report as a text file.
with open(os.path.join(log_dir, "classification_report.txt"), 'w') as f:
    f.write(report)

# Close Tensorboard writer
writer.close()
print("Finished training!")