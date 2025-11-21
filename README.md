### **Name : Prudhvi VVR Ajay Sreenivas**

### **Roll No. : DA25M023**

---

# **Predicting LLM-Judge Fitness Scores from Metric Definitions and Response Behavior**

This repository contains the implementation and modeling workflow for predicting evaluation metric scores of AI-generated healthcare responses. The project explores multi-output regression using embedding-based feature engineering, negative sampling, neural networks, and calibration techniques.

---

## **Project Overview**

Large Language Models (LLMs) are often evaluated using a large set of metric-based scores.
The goal of this project is to **predict ~145 evaluation metric scores** for a given:

* **System prompt**
* **User healthcare query**
* **LLM-generated response**

The dataset includes:

* **5,000 training samples**
* **3,638 test samples**

Each evaluation score is in the **0–10 range**.

---

## **Dataset Structure**

Each example contains:

* `<SYS>` System prompt
* `<USER>` User query
* `<RESP>` Generated AI response

These were **not merged**, but encoded with special tokens to help the model understand roles.

---

## **Data Preprocessing**

Key preprocessing steps:

* Kept all text parts separate using tokens: `<SYS>`, `<USER>`, `<RESP>`
* Converted missing values to empty strings
* Used **multilingual-e5-large** to compute **1024-dimensional embeddings** for:

  * Full text (SYS + USER + RESP)
  * Names of all metrics
* Embedding generation performed in batches of 64 to manage GPU memory

---

## **Data Augmentation via Negative Sampling**

To avoid overfitting (only 5k samples), negative samples expanded the dataset to ~20k.

Three negative strategies were used:

1. **Random Shuffling** of text embeddings
2. **Gaussian Noise Injection** (σ = 0.6)
3. **Metric Shuffling** to break semantic alignment

All negative samples were assigned low scores (0–2).
The best performance came from a **3:1 negative-to-positive ratio**.

---

## **Exploratory Data Analysis**

* Performed stratified splitting using rounded labels
* Verified embedding clusters to ensure semantic quality
* Ensured score distribution consistency across folds

---

## **Feature Engineering**

Each data point converts into a **4097-dimensional feature vector**:

| Feature Type                              | Dimensions |
| ----------------------------------------- | ---------- |
| Concatenation of metric + text embeddings | 2048       |
| Absolute Difference                       | 1024       |
| Element-wise Product                      | 1024       |
| Cosine Similarity                         | 1          |
| **Total**                                 | **4097**   |

This multi-view representation captures similarity, difference, and alignment.

---

## **Model Architecture**

A custom feed-forward neural network:

```
4097 → 1024 → 512 → 128 → 1
```

* Activation: ReLU
* Dropout: 20% after first two layers
* Implemented in PyTorch

Chosen for its balance of expressive power and generalization.

---

## **Training Pipeline**

**5-fold stratified cross-validation**
Each fold trained for **20 epochs** with:

* Optimizer: **AdamW**
* Learning rate: **0.001**
* Weight decay: **1e-5**
* Batch size: **256**
* Loss: **MSE**

Best model per fold saved.

---

## **Hacks & Optimizations**

### Calibration Layer

A **Ridge Regression** layer (α = 1.0) was trained on outputs to correct bias.
**RMSE improved from 3.13 → 3.08.**

### Embedding Caching

Saved embedding matrices to disk to avoid recomputation.

### Prediction Clipping

Ensured predictions remain in the valid range [0, 10].

### Model Averaging

All 5 folds were averaged for stable test predictions.

---

## **Results**

| Fold                  | Best RMSE |
| --------------------- | --------- |
| 1                     | 3.05      |
| 2                     | 3.12      |
| 3                     | 3.18      |
| 4                     | 3.09      |
| 5                     | 3.21      |
| **Average**           | **3.13**  |
| **After Calibration** | **3.08**  |

The small fold-to-fold variance shows strong generalization.

---

## **Conclusion**

This project demonstrates that:

* Combining **dual embeddings** (text + metric)
* Using **negative sampling**
* Designing **multi-view feature vectors**
* Applying **post-hoc calibration**

leads to a strong predictive model for automated quality scoring.

The final calibrated RMSE of **3.08** shows that the system captures meaningful semantic relationships between prompts, responses, and evaluation metrics.

