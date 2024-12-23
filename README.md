# 🖥️ End-to-End GPU-Optimized Pipeline for News Recommendation Systems 🚀

## ✨ Project Overview
This project showcases the development of an **end-to-end GPU-optimized recommendation system** using NVIDIA GPU libraries to preprocess, analyze, and train models on the **Microsoft MIND dataset**. By reformatting and optimizing the dataset for deep learning frameworks, we drastically reduced preprocessing time from days on CPUs to hours on GPUs. 

Key features include:
- GPU-accelerated preprocessing using **cuDF**, **cuPY**, and **NVTabular**.
- Training a **Deep Learning Recommendation Model (DLRM)** with **HugeCTR**.
- Achieving a **ROC-AUC score of 0.644** on the test set.

---

## 📚 Dataset Details
- **Dataset**: Microsoft MIND (160K news articles, 15M impression logs).
- **Data Fields**:
  - **News**: Category, subcategory, title, and abstract.
  - **Impressions**: Click and non-click events with historical user behavior.
- **Challenges**: Preprocessing historical and impression data, class imbalance, and high memory requirements.

---

## 🛠️ Design & Implementation
### 📐 Design
- **Tools**: AWS EC2, LambdaLabs, NVIDIA A100 GPUs.
- **Storage**: Artifacts and datasets stored in **Parquet format** on **AWS S3**.
- **Libraries**: cuDF, cuPY, NVTabular, Dask, HugeCTR.

### 🏗️ Implementation
1. **Preprocessing**:
   - Filtered unnecessary columns.
   - Label-encoded categorical data for memory efficiency.
   - Unrolled history and impression columns for feature engineering.
   - Balanced the dataset with **negative sampling**.
2. **Feature Engineering**:
   - **Time-based features** (e.g., day of the week, hour).
   - **Count encoding** for categorical frequencies.
   - **Target encoding** based on target variable averages.
3. **Model Training**:
   - Model: **DLRM** with embedding layers and fully connected layers.
   - Optimized using custom CUDA kernels, reducing training time by 10%.

---

## 📊 Results & Metrics
- **Evaluation Metric**: ROC-AUC.
- **Performance Gains**:
  - Preprocessing steps were **7–12x faster on GPUs** compared to CPUs.
  - **Training time reduced by 240x** using HugeCTR.

| **Dataset Version**        | **Train AUC** | **Test AUC** |
|-----------------------------|---------------|--------------|
| Time-Based Features         | 0.7433        | 0.6615       |
| Count + Target Encoded      | 0.7239        | 0.6543       |

| **Operation**              | **CPU Time** | **GPU Time** | **Acceleration** |
|----------------------------|--------------|--------------|-------------------|
| Unroll History             | 8 hours      | 1 hour       | 8x               |
| Feature Engineering (Count)| 24 hours     | 2 hours      | 12x              |
| Training                   | 60 hours     | 15 minutes   | 240x             |

---

## 🌟 Key Contributions
- Developed a GPU-optimized pipeline for preprocessing and training.
- Improved dataset accessibility for low-resource settings.
- Highlighted the importance of efficient infrastructure in large-scale AI projects.

---

## 🛠️ Tools & Technologies
- **Hardware**: NVIDIA A100 and L4 GPUs.
- **Libraries**: cuDF, cuPY, NVTabular, Dask, HugeCTR.
- **Cloud**: AWS, LambdaLabs.

---

## 🗺️ Future Scope
- Incorporating textual data (e.g., titles and abstracts) for feature engineering.
- Exploring advanced techniques to address class imbalance.
- Open-sourcing scripts and workflows for reproducibility.


---

## 🔗 References
1. [MIND Dataset and News Recommendation](https://arxiv.org/abs/2108.05314)
2. [NVIDIA Merlin Framework](https://developer.nvidia.com/nvidia-merlin)
3. [HugeCTR Framework](https://arxiv.org/abs/2110.11051)

---
