# # 🩺 Breast Cancer Classification Using Supervised Learning

## 📌 Objective

This project aims to evaluate and compare five popular supervised learning classification algorithms using the Breast Cancer Wisconsin dataset from the `sklearn` library. The goal is to predict whether a tumor is malignant or benign based on several features computed from digitized images of a breast mass.

---

## 📂 Dataset

- Source: `sklearn.datasets.load_breast_cancer()`
- Instances: 569
- Features: 30 (numerical), e.g., mean radius, mean texture, mean area, etc.
- Target: Binary (`0` = malignant, `1` = benign)

---

## 🧼 Preprocessing Steps

1. **Converted data to DataFrame**
2. **Checked for missing values** (none found)
3. **Capped outliers** using the IQR method
4. **Scaled features** using StandardScaler for better model performance

---

## 🧠 Classification Algorithms Used

### 1. Logistic Regression
- A linear model that estimates class probabilities using the logistic function.
- ✔️ Works well with linearly separable features, fast and interpretable.

### 2. Decision Tree
- A flowchart-like structure that splits data based on feature thresholds.
- ✔️ Easy to interpret; ⚠️ risk of overfitting.

### 3. Random Forest
- An ensemble of decision trees using bagging and feature randomness.
- ✔️ Robust to overfitting and performs well without much tuning.

### 4. Support Vector Machine (SVM)
- Finds the optimal margin hyperplane between classes.
- ✔️ Excellent for high-dimensional, well-separated data.

### 5. k-Nearest Neighbors (k-NN)
- Classifies based on the majority label among the k closest points.
- ✔️ Simple; ⚠️ sensitive to noise and feature scaling.

---

## 📊 Results Summary

| Classifier              | Accuracy (%) |
|-------------------------|--------------|
| **Logistic Regression** | **98.25%** ✅ |
| Random Forest           | 96.49%       |
| SVM                     | 96.49%       |
| k-NN                    | 95.61%       |
| Decision Tree           | 94.74% ❌     |

- **Best Performer**: Logistic Regression
- **Worst Performer**: Decision Tree (still good)

### 🧪 Evaluation Metrics
Each model was evaluated using:
- ✅ **Accuracy**
- ✅ **Confusion Matrix**
- ✅ **Precision, Recall, F1-score**

---

## 📎 Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
