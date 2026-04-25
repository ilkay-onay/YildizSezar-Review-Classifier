# 🌟 YıldızSezar: Turkish E-Commerce Review Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilkay-onay/YildizSezar-Review-Classifier/blob/main/YildizSezar_Demo.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ilkayO/YildizSezar-Demo)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

This repository contains the official implementation of **YıldızSezar**, my B.Sc. Thesis Project: **"Multi-class Sentiment Analysis with ConvBERT on Turkish E-Commerce Reviews"**, which was successfully published in a peer-reviewed journal.

🔗 **[Read the Full Paper Here](#)** <!-- TODO: Makale linkiniz oluşunca buraya ekleyin -->
🤖 **[Try the Live Model on Hugging Face](https://huggingface.co/spaces/ilkayO/YildizSezar-Demo)** 
📊 **[Explore the Dataset on Hugging Face](https://huggingface.co/datasets/ilkayO/yildizsezar-turkish-reviews)** 

## 📌 Project Overview
The objective of **YıldızSezar** is to automatically predict 1-to-5 star ratings directly from morphologically complex Turkish customer reviews. 

To combat severe class imbalance (skewed towards 5-star and 1-star), I engineered a scalable synthetic data generation pipeline using **LLaMA-8B-DPO**, generating over **900,000 synthetic review samples**. 

The final **ConvBERT-based classifier** achieved an accuracy of **89.96%** and a macro F1-score of **0.799**.

## 🚀 Key Features & Production Readiness
* **Advanced Deep Learning Models:** Fine-tuned ConvBERT, DistilBERT, and ELECTRA.
* **Synthetic Data Augmentation:** Solved extreme class imbalance using open-source LLMs.
* **Gradio & Flask Web UIs:** Built lightweight web interfaces for real-time model inference and probability visualization.
* **Production Performance Testing:** Implemented scripts to measure Latency (ms) and Throughput (QPS) for production environments.
* **Reliability & Calibration:** Analyzed Expected Calibration Error (ECE) and generated reliability diagrams to ensure the model's confidence scores are trustworthy.

## 📊 Model Evaluation & Calibration
*(The figures below demonstrate the model's performance on the test set and its confidence calibration.)*

<p align="center">
  <img src="evaluation/precision_recall_curve_final_v3.png" width="45%" />
  <img src="evaluation/reliability_diagram_multiclass.png" width="45%" />
</p>

## 📂 Repository Structure
* `/data_processing`: Scripts for data cleaning, HTML unescaping, and dataset splitting.
* `/training`: PyTorch/Transformers training loops, including Distributed Data Parallel (DDP).
* `/inference`: Scripts for basic model loading and terminal inferences.
* `/evaluation`: Advanced evaluation scripts (Latency, Throughput, Expected Calibration Error, PR curves).
* `/web_app`: Local Flask-based Web Interface for live demonstrations.

## 💻 Quick Start

You can load the model directly via Hugging Face `transformers`:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "ilkayO/yildizsezar-convbert" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

review = "Biraz gecikmeli aktarıyor görüntüyü ama onun dışında fiyat performans ürünü."
inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()

# Assuming id2label is properly configured in HF
print(f"Predicted Star Rating: {model.config.id2label[str(predicted_class_id)]}")
