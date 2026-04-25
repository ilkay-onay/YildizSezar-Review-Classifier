# 🌟 YıldızSezar: Turkish E-Commerce Review Classifier

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilkay-onay/YildizSezar-Review-Classifier/blob/main/YildizSezar_Demo.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ilkayO/YildizSezar-Demo)

This repository contains the official implementation of **YıldızSezar**, my B.Sc. Thesis Project: **"Multi-class Sentiment Analysis with ConvBERT on Turkish E-Commerce Reviews"**, which was successfully published in the *International Journal of 3D Printing Technologies and Digital Industry*.

🔗 **[Read the Full Peer-Reviewed Paper Here](https://dergipark.org.tr/tr/pub/ij3dptdi/article/1732179)**
🤖 **[Access the Trained Model on Hugging Face](https://huggingface.co/ilkayO/yildizsezar-convbert)**
📊 **[Explore the Dataset on Hugging Face](https://huggingface.co/datasets/ilkayO/yildizsezar-turkish-reviews)**

## 📌 Project Overview
The objective of **YıldızSezar** is to automatically predict 1-to-5 star ratings directly from morphologically complex Turkish customer reviews. 

Real-world e-commerce data often suffers from extreme class imbalance (heavily skewed towards 5-star and 1-star). To combat this, I engineered a scalable synthetic data generation pipeline using **LLaMA-8B-DPO**, generating over **900,000 synthetic review samples** for minority classes. 

The final **ConvBERT-based classifier** achieved an accuracy of **89.96%** and a macro F1-score of **0.799**, significantly outperforming traditional ML baselines.

## 🚀 Key Features & Production Readiness
* **Advanced Deep Learning Models:** Fine-tuned ConvBERT, DistilBERT, and ELECTRA architectures.
* **Synthetic Data Augmentation:** Solved extreme class imbalance using open-source LLMs.
* **Flask & Gradio Web UI:** Integrated lightweight web interfaces for real-time model inference and probability visualization.
* **Production Performance Testing:** Implemented scripts to measure Latency (ms) and Throughput (QPS) for deployment environments.
* **Reliability & Calibration:** Analyzed Expected Calibration Error (ECE) and generated reliability diagrams to ensure the model's confidence scores are trustworthy.

## 📂 Repository Structure
* `/data_processing`: Scripts for data cleaning, HTML unescaping, and dataset splitting.
* `/training`: PyTorch/Transformers training loops, including Distributed Data Parallel (DDP).
* `/inference`: Scripts for basic model loading and terminal inferences.
* `/evaluation`: Advanced evaluation scripts (Latency, Throughput, Expected Calibration Error, Precision-Recall curves).
* `/web_app`: Flask-based Web Interface for local testing.

## 📊 Model Evaluation & Calibration
Below are the rigorous evaluation metrics validating the model's statistical reliability and precision across all 5 classes.
<div style="display: flex; justify-content: space-between;">
  <img src="evaluation/precision_recall_curve_final_v3.png" width="48%">
  <img src="evaluation/reliability_diagram_multiclass.png" width="48%">
</div>

## 💻 Quick Start (Local Inference)
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
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

# Assumes config.id2label is properly configured (e.g., 0 -> "1 Yıldız")
print(f"Predicted Star Rating: {model.config.id2label[str(predicted_class)]}")
```
