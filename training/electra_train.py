import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import random
from transformers import EarlyStoppingCallback
from datetime import datetime

# Step 1: Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Step 2: Load both datasets
# Original dataset
df1 = pd.read_csv('hb.csv')

# Additional dataset
df2 = pd.read_csv('ecommerce_review_dataset.csv')

# Step 3: Preprocess datasets
# Drop missing values
df1 = df1.dropna(subset=['Review', 'Rating (Star)'])
df2 = df2.dropna(subset=['review', 'star'])

# Standardize column names for merging
df1 = df1.rename(columns={"Review": "review", "Rating (Star)": "star"})
df2 = df2[['review', 'star']]  # Select only review and star columns

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Convert reviews: Remove extra spaces but retain original casing
combined_df['review'] = combined_df['review'].str.strip()


# Convert star ratings to integers (if needed)
combined_df['star'] = combined_df['star'].astype(int)

# Step 4: Prepare features (X) and labels (y)
X = combined_df['review'].tolist()  # Features: reviews
y = combined_df['star'].values  # Labels: star ratings

# Map star ratings to zero-based indices (e.g., 1-5 -> 0-4)
label_map = {rating: i for i, rating in enumerate(sorted(set(y)))}
y_mapped = np.array([label_map[rating] for rating in y])

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=seed)

# Step 5: Initialize the ELECTRA tokenizer
tokenizer = ElectraTokenizer.from_pretrained('dbmdz/electra-base-turkish-mc4-cased-discriminator')

# Custom Dataset class for ELECTRA
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 6: Create Dataset objects for training and testing
train_dataset = ReviewDataset(X_train, y_train, tokenizer)
test_dataset = ReviewDataset(X_test, y_test, tokenizer)

# Step 7: Load the pre-trained ELECTRA model for sequence classification
model = ElectraForSequenceClassification.from_pretrained(
    'dbmdz/electra-base-turkish-mc4-cased-discriminator',
    num_labels=len(label_map)  # Number of unique star ratings
)

# Step 8: Define training arguments
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging_dir = f"./logs/{current_time}"
output_dir = f"./results/{current_time}"

training_args = TrainingArguments(
    output_dir=output_dir,              
    num_train_epochs=4,                 
    per_device_train_batch_size=32,    
    per_device_eval_batch_size=32,      
    warmup_ratio=0.1,                   # warmup arttirildi
    weight_decay=0.02,                  # generalizasyon icin agirlik eritme artirildi
    learning_rate=3e-5,                 # ogrenme hizi azaltildi
    lr_scheduler_type="cosine",         # faydali diye onerildi?
    max_grad_norm=1.0,                  # gradyantta stabilize icin yardimci olan bir parametre
    logging_dir=logging_dir,            
    logging_steps=10,                   
    evaluation_strategy="epoch",        
    save_strategy="epoch",              
    save_total_limit=3,                 
    load_best_model_at_end=True,        
    metric_for_best_model="eval_loss",  
    seed=seed,                          
    report_to="tensorboard",            
    fp16=torch.cuda.is_available(),     
    gradient_accumulation_steps=2
)


# Step 9: Define the compute_metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Step 10: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Step 11: Train the model
print("Starting training...")
trainer.train()

# Step 12: Evaluate the model
print("Evaluating model...")
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)

# Step 13: Save the trained model
model_dir = f'./final_model_{current_time}'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Model and tokenizer saved at {model_dir}!")

# Step 14: Test the model on the test set
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Additional metrics: classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_labels))

# Test accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")