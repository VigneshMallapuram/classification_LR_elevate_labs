# Task 4: Classification with Logistic Regression

## Objective

The objective of this project is to build a binary classification model using **Logistic Regression**. The model predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using various cell nucleus measurements from the Breast Cancer Wisconsin Dataset.

---

## Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

The dataset contains measurements computed from digitized images of fine needle aspirates (FNA) of breast masses.

* **Total Samples:** 569
* **Features:** 30 numerical features
* **Target Variable:** `diagnosis`

  * **M** → Malignant (1)
  * **B** → Benign (0)

Before training the model:

* The `id` column was removed because it is not useful for prediction.
* The `Unnamed: 32` column was removed because it contains only null values.
* The diagnosis labels were converted into numerical values.

---

## Tools and Libraries Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

---

## Steps Performed

### 1. Import Libraries

Imported all the required libraries for data preprocessing, model training, visualization, and evaluation.

---

### 2. Load the Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("data.csv")
```

---

### 3. Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns (`id` and `Unnamed: 32`)
* Converted target labels:

  * M → 1
  * B → 0
* Separated features (`X`) and target (`y`)

---

### 4. Split the Dataset

The dataset was divided into training and testing sets.

* Training Data: **80%**
* Testing Data: **20%**

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Using `stratify=y` ensures that both training and testing datasets maintain the same proportion of benign and malignant cases.

---

### 5. Feature Scaling

Feature scaling was performed using **StandardScaler**.

Logistic Regression performs better when all features are on a similar scale.

The standardization formula is:

[
z=(x-mu)/sigma
]

where:

* (x) = Feature value
* (mu) = Mean
* (sigma) = Standard deviation

---

### 6. Train the Logistic Regression Model

A Logistic Regression classifier was trained using the standardized training data.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

### 7. Sigmoid Function

Logistic Regression uses the **Sigmoid Function** to convert linear outputs into probabilities between 0 and 1.

[
sigma(z)={1}/{1+e^{-z}}
]

where

[
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
]

If the probability is greater than or equal to **0.5**, the sample is classified as **Malignant**; otherwise, it is classified as **Benign**.

---

### 8. Model Prediction

The trained model predicts:

* Class labels
* Prediction probabilities

These probabilities are later used to compute the ROC Curve and ROC-AUC score.

---

### 9. Model Evaluation

The classifier was evaluated using the following metrics.

#### Accuracy

Measures the percentage of correctly classified samples.

[
Accuracy={TP+TN}/{TP+TN+FP+FN}
]

---

#### Precision

Measures how many predicted positive cases are actually positive.

[
Precision={TP}/{TP+FP}
]

---

#### Recall

Measures how many actual positive cases are correctly identified.

[
Recall={TP}/{TP+FN}
]

---

#### Confusion Matrix

The confusion matrix summarizes the model's predictions by displaying:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

---

#### ROC Curve

The Receiver Operating Characteristic (ROC) Curve shows the trade-off between:

* True Positive Rate
* False Positive Rate

A curve closer to the top-left corner indicates better performance.

---

#### ROC-AUC Score

The Area Under the ROC Curve (ROC-AUC) measures the model's ability to distinguish between the two classes.

* **1.0** → Perfect classifier
* **0.5** → Random guessing

A higher ROC-AUC score indicates a better model.

---

### 10. Threshold Tuning

By default, Logistic Regression uses a threshold of **0.5** for classification.

This project also demonstrates threshold tuning by changing the threshold to **0.4**.

Lowering the threshold generally:

* Increases Recall
* May decrease Precision

Threshold tuning helps balance false positives and false negatives depending on the application.

---

## Results

The Logistic Regression model achieved excellent performance on the Breast Cancer Wisconsin Dataset.

Typical output:

* Accuracy: 96.4%
* Precision: 97.5%
* Recall: 92.8%
* ROC-AUC: 99.6%

The exact values may vary slightly depending on the train-test split.

---

## Output Files

* `data.csv`
* `logistic_regression.py`
* `confusion_matrix.png`
* `roc_curve.png`
* `README.md`

---

## Concepts Learned

Through this project, the following concepts were learned:

* Binary Classification
* Logistic Regression
* Feature Scaling
* Train-Test Split
* Standardization
* Sigmoid Function
* Confusion Matrix
* Precision
* Recall
* ROC Curve
* ROC-AUC Score
* Threshold Tuning

---

## Conclusion

This project demonstrates the complete workflow of building a binary classification model using Logistic Regression. Starting from data preprocessing and feature scaling to model training, evaluation, and threshold tuning, the project provides a practical understanding of binary classification and performance evaluation techniques. The Logistic Regression model achieved high accuracy and an excellent ROC-AUC score, making it an effective baseline algorithm for medical diagnosis and other binary classification problems.
