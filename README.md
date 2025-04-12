# 🚀 Logistic Regression Project with Streamlit Deployment

This repository showcases a complete machine learning pipeline using **Logistic Regression**, including:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Building and Evaluation
- Interpretation of results
- Deployment using **Streamlit**

---

## 📁 Project Structure

```bash
├── data/
│   └── dataset.csv
├── images/
│   └── step1.png
│   └── step2.png
├── model/
│   └── logistic_model.pkl
├── app.py
├── eda_notebook.ipynb
├── requirements.txt
└── README.md
1️⃣ Data Exploration
We start with loading the dataset and performing Exploratory Data Analysis.

Summary statistics

Histograms and Boxplots

Correlation matrix

2️⃣ Data Preprocessing
Impute missing values using median/most frequent strategy.

Encode categorical variables with LabelEncoder or OneHotEncoder.

from sklearn.preprocessing import LabelEncoder
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

3️⃣ Model Building
Using LogisticRegression from scikit-learn:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
4️⃣ Model Evaluation
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

ROC Curve visualization
from sklearn.metrics import roc_curve, roc_auc_score

5️⃣ Interpretation
Model coefficients help us understand feature importance:

coeff_df = pd.DataFrame(model.coef_.T, index=feature_names, columns=["Coefficient"])

🛠 Requirements:
Includes:
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit



📬 Contact
For feedback or queries, feel free to connect!

