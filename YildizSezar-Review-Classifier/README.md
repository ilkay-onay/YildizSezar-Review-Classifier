# 🌟 YıldızSezar: Turkish E-Commerce Review Classifier

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)

This repository contains the official implementation of **YıldızSezar**, my B.Sc. Thesis Project: **"Multi-class Sentiment Analysis with ConvBERT on Turkish E-Commerce Reviews"**, which was successfully published in a peer-reviewed journal.

🔗 **[Read the Full Paper Here](#)** <!-- TODO: Makale linkini ekle -->
🤖 **[Try the Live Model on Hugging Face](#)** <!-- TODO: HF Model linkini ekle -->
📊 **[Explore the Dataset on Hugging Face](#)** <!-- TODO: HF Dataset linkini ekle -->

## 📌 Project Overview
The objective of **YıldızSezar** is to automatically predict 1-to-5 star ratings directly from morphologically complex Turkish customer reviews. 

To combat severe class imbalance, I engineered a scalable synthetic data generation pipeline using **LLaMA-8B-DPO**, generating over **900,000 synthetic review samples**. 

The final **ConvBERT-based classifier** achieved an accuracy of **89.96%** and a macro F1-score of **0.799**.

## 🚀 Key Features & Production Readiness
* **Advanced Deep Learning Models:** Fine-tuned ConvBERT, DistilBERT, and ELECTRA.
* **Synthetic Data Augmentation:** Solved extreme class imbalance using open-source LLMs.
* **Flask Web UI:** Integrated a lightweight web interface for real-time model inference and probability visualization.
* **Production Performance Testing:** Implemented scripts to measure Latency (ms) and Throughput (QPS) for production environments.
* **Reliability & Calibration:** Analyzed Expected Calibration Error (ECE) and generated reliability diagrams to ensure the model's confidence scores are trustworthy.

## 📂 Repository Structure
* `/data_processing`: Scripts for data cleaning, HTML unescaping, and dataset splitting.
* `/training`: PyTorch/Transformers training loops, including Distributed Data Parallel (DDP).
* `/inference`: Scripts for basic model loading and terminal inferences.
* `/evaluation`: Advanced evaluation scripts (Latency, Throughput, Expected Calibration Error, Precision-Recall curves).
* `/web_app`: Flask-based Web Interface for live demonstrations.

## 💻 Quick Start (Web UI)
To run the local web interface and test the model in your browser:

```bash
cd web_app
python app.py
```
*Then navigate to `http://127.0.0.1:5000` in your browser.*
