# SemEval 2025 Task 9: Food Hazard Detection

## Overview
This project is part of **SemEval 2025 Task 9**, focusing on **food hazard detection** using **machine learning & NLP**.  
We fine-tune a **RoBERTa transformer model** on **AWS SageMaker** to classify potential food hazards from textual data.

## Technologies Used
- **Python** (Data Processing & Model Training)
- **Pandas** (Data Handling)
- **Scikit-Learn** (Data Preprocessing & Metrics)
- **Hugging Face Transformers** (Pretrained RoBERTa Model)
- **PyTorch** (Deep Learning)
- **AWS SageMaker** (Model Training & Deployment)

## Dataset & Preprocessing
- **Dataset:** https://food-hazard-detection-semeval-2025.github.io/ 
- Data is **preprocessed, tokenized, and split** into training & test sets.  
- Labels are **encoded** using `LabelEncoder`.

## Model Training on AWS SageMaker
- We fine-tune a **RoBERTa-based transformer model** using **AWS SageMaker**.
- The training process leverages **SageMaker’s GPU-accelerated instances** for faster convergence.
- **Evaluation metric:** **F1 Score** (Used to assess model performance).  
