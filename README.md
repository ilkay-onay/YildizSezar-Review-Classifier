# 🌟 YıldızSezar: Turkish E-Commerce Review Classifier

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)
![Status](https://img.shields.io/badge/status-production_ready-success)

This repository contains the official implementation of **YıldızSezar**, my B.Sc. Thesis Project: **"Multi-class Sentiment Analysis with ConvBERT on Turkish E-Commerce Reviews"**, which was successfully published in a peer-reviewed journal.

🔗 **[Read the Full Paper Here](#)** <!-- TODO: Makale linkini ekle -->
🤗 **[Try the Live Model on Hugging Face](#)** <!-- TODO: Hugging Face linkini ekle -->
📊 **[Explore the Dataset on Kaggle](#)** <!-- TODO: Kaggle linkini ekle -->

## 📌 Project Overview
The objective of **YıldızSezar** is to automatically predict 1-to-5 star ratings directly from morphologically complex Turkish customer reviews. 

To combat severe class imbalance, I engineered a scalable synthetic data generation pipeline using **LLaMA-8B-DPO**, generating over **900,000 synthetic review samples**. 

The final **ConvBERT-based classifier** achieved an accuracy of **89.96%** and a macro F1-score of **0.799**, significantly outperforming traditional ML baselines.

## 🚀 Key Features
* **Advanced Deep Learning Models:** Fine-tuned ConvBERT, DistilBERT, and ELECTRA architectures.
* **Synthetic Data Augmentation:** Solved extreme class imbalance using open-source LLMs.
* **Enterprise-Grade Training Loop:** Implemented Custom Class Weights, Early Stopping, Cosine Learning Rate Schedulers, and TensorBoard logging.
* **Distributed Training:** Optimized training scripts for Multi-GPU setups (DDP).

## 📂 Repository Structure
* `/data_processing`: Scripts for data cleaning, HTML unescaping, and dataset splitting.
* `/training`: PyTorch/Transformers training loops, including Distributed Data Parallel (DDP).
* `/inference`: Scripts for loading the trained model and running inferences.

## 💻 Quick Start (Inference)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "your-huggingface-username/yildizsezar-convbert" # TODO: Burayi guncelle
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

review = "Biraz gecikmeli aktariyor goruntuyu ama onun disinda fiyat performans urunu."
inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

print(f"Predicted Star Rating: {predicted_class + 1} Stars")
```
