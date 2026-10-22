# Predictive Maintenance System using Machine Learning and Flask

A web application that predicts machine failure from sensor readings using a Random Forest model. Built on the AI4I 2020 Predictive Maintenance dataset (10,000 records) and served with Flask.

## Features

- **Single prediction:** enter sensor readings and get the failure prediction and probability.
- **Batch prediction:** upload a CSV of machine readings and get predictions for every row.
- **Dashboard:** visual overview of the data and model results.
- **About page:** project and model background.

## Project Structure

```
.
├── app.py                      # Flask application
├── predictive.csv              # Dataset
├── requirements.txt            # Python dependencies
├── static/
│   └── css/style.css           # Styling
├── templates/
│   ├── base.html               # Shared layout
│   ├── predict.html            # Single prediction form
│   ├── result.html             # Prediction result
│   ├── batch.html              # Batch upload page
│   ├── batch_result.html       # Batch results
│   ├── dashboard.html          # Dashboard
│   └── about.html              # About page
└── Predictive Maintenance System using Machine Learning and Flask.ipynb   # EDA, model selection, tuning
```

## Machine Learning Workflow

1. **Data cleaning:** no missing values or duplicates; outliers in rotational speed and torque capped with the IQR rule.
2. **Feature engineering:**
   - `Temp_diff` = process temperature − air temperature
   - `Power` = torque × angular velocity
   - `Wear_rate` = tool wear ÷ rotational speed
3. **Preprocessing:** MinMax scaling for numeric features, one-hot encoding for product `Type` (L, M, H).
4. **Class imbalance:** only about 3.4% of records are failures, so evaluation focuses on precision, recall, F1 and PR-AUC rather than accuracy.
5. **Model selection:** Logistic Regression, Decision Tree and Random Forest were compared; Random Forest was chosen by F1-score and tuned with `RandomizedSearchCV` (using `class_weight='balanced_subsample'`).

### Results (20% stratified test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.966 | 0.00 | 0.00 | 0.00 | 0.846 |
| Decision Tree | 0.984 | 0.78 | 0.74 | 0.76 | 0.864 |
| Random Forest | 0.991 | 0.93 | 0.79 | 0.86 | 0.975 |
| Random Forest (tuned) | 0.990 | 0.91 | 0.78 | 0.84 | 0.966 |

## Tech Stack

Python, Flask, scikit-learn, pandas, NumPy, Matplotlib / Seaborn, HTML, CSS

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<varshanaut>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Input Features

| Feature | Description |
|---|---|
| Type | Product quality variant: L, M or H |
| Air temperature [K] | Ambient temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Spindle speed |
| Torque [Nm] | Torque applied |
| Tool wear [min] | Cumulative tool usage time |

For batch prediction, the uploaded CSV must contain these columns with the same names.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), UCI Machine Learning Repository.

## Notes

- Flask's built-in server is for development only; use gunicorn or a similar server in production.
- Recall on the failure class is about 0.78, so some failures are still missed. Lowering the decision threshold catches more failures at the cost of more false alarms.

## Author

Varsha Praveen