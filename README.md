# 🛡️ Early Phishing Detection Web App

A **Streamlit-powered** web application to detect potential phishing accounts using transactional data and logistic regression. Upload a CSV of transactions and receive actionable insights, model evaluation, and detection results.

---

## 📌 Features

- 📁 Upload and process your transaction CSV file  
- 🧼 Data cleaning and feature engineering  
- ⚖️ Automatic handling of class imbalance (upsampling)  
- 🤖 Logistic Regression-based phishing prediction  
- 📊 Performance metrics: Accuracy, Precision, Recall, F1-Score  
- 📉 Feature importance visualization  
- 📋 Comparison with Node2Vec & Trans2Vec models  
- 🔍 Check predictions for individual sender addresses  

---

## 📂 Input CSV Format

Your CSV file should contain the following columns:

| Column Name  | Description                             |
|--------------|-----------------------------------------|
| `from`       | Sender address                          |
| `amount`     | Transaction amount                      |
| `timestamp`  | UNIX timestamp of the transaction       |
| `date`       | Human-readable date of the transaction  |
| `fromIsPhi`  | Label: 1 for phishing, 0 for legitimate |

> ⚠️ Columns like `Unnamed: 0` are auto-ignored if present.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/phishing-detector.git
cd phishing-detector
```
### 2: Create and Activate Virtual Environment

#### 🖥️ For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
### 3: Install Dependencies

```bash
pip install -r requirements.txt
```
### 3: Run the Streamlit app

```bash
streamlit run app.py
```
## 🔎 How It Works

1. **Data Upload**: Accepts CSV of transaction records.  
2. **Feature Engineering**: Calculates aggregated stats per sender (avg amount, skewness, tx count, etc.).  
3. **Balancing**: Upsamples phishing transactions to counter class imbalance.  
4. **Training**: Fits a logistic regression model.  
5. **Prediction**: Predicts phishing accounts on test data.  
6. **Evaluation**: Displays confusion matrix and metrics.  
7. **Visualization**: Shows feature importance.  
8. **Query**: Allows sender-specific phishing prediction.

---

## 📊 Sample Model Comparison

| Model                        | Accuracy | Precision | Recall | F1 Score |
|-----------------------------|----------|-----------|--------|----------|
| Logistic Regression (Ours)  | 0.XX     | 0.XX      | 0.XX   | 0.XX     |
| Node2Vec                    | 0.72     | 0.53      | 0.22   | 0.31     |
| Trans2Vec                   | 0.81     | 0.61      | 0.31   | 0.41     |

> 📝 Replace `0.XX` with the actual results after running the model.

---

## 📦 Dependencies

- `streamlit`  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  

To generate your own `requirements.txt`, run:

```bash
pip freeze > requirements.txt
```
## 🙋‍♂️ Author

**Sumitra Saha**  
GitHub: [@SumitraSaha04](https://github.com/SumitraSaha04)  

**Raj Prabhakar**  
GitHub: [@rajprabhakar007](https://github.com/rajprabhakar007)  
