# 🧥 Fashion MNIST Pro: Advanced Custom ResNet & Explainable AI Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Computer%20Vision-success)

## 📌 Project Overview
This project is an end-to-end, production-ready Machine Learning pipeline designed to classify clothing items from the Fashion MNIST dataset. Moving beyond simple sequential models, this repository implements a **Custom Residual Network (ResNet)** built from scratch and integrates **Explainable AI (XAI)** techniques to ensure model transparency and trustworthiness.

The architecture is divided into a professional 3-stage MLOps pipeline: Data Ingestion, Model Training, and Evaluation/Inference.

## ✨ Key Features
* **Custom ResNet Architecture:** Implements custom `ResidualBlock` layers with skip connections to mitigate the vanishing gradient problem, allowing for deeper, more stable feature extraction.
* **Explainable AI (Grad-CAM):** Goes beyond accuracy metrics by generating Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps. This proves *where* the model is looking when making a prediction.
* **Global Average Pooling:** Replaces traditional dense flattening to drastically reduce parameter count and prevent overfitting while retaining spatial hierarchies.
* **Modular MLOps Pipeline:** Code is strictly separated into extraction (ETL), training, and evaluation scripts, mirroring real-world enterprise engineering standards.

## 🗂️ Repository Structure
```text
fashion_mnist_pro/
│
├── data/                       # Serialized numpy arrays (Git Ignored)
├── models/                     # Saved .keras production models 
├── outputs/                    # Generated confusion matrices and Grad-CAM heatmaps
│
├── 01_data_pipeline.py         # Node 1: Downloads, normalizes, and serializes data
├── 02_train_pipeline.py        # Node 2: Compiles the ResNet and handles model training
├── 03_eval_pipeline.py         # Node 3: Generates classification reports and XAI visuals
│
├── fashion_mnist_pro.ipynb     # Presentation notebook for stakeholders/reviewers
├── requirements.txt            # Environment dependencies
└── README.md                   # Project documentation
