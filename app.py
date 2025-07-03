import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
import seaborn as sns

st.set_page_config(page_title="Early Phishing Detection", layout="wide")
st.title("Early Phishing Detection Web App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully.")

    # Step 1: Preprocessing
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Step 2: Feature Engineering
    grouped = df.groupby("from")
    features = grouped["amount"].agg([
        ("tx_count", "count"),
        ("avg_amount", "mean"),
        ("max_amount", "max"),
        ("min_amount", "min"),
        ("std_amount", "std"),
        ("skewness", lambda x: x.skew()),
        ("kurtosis", lambda x: x.kurtosis())
    ]).fillna(0)

    features["first_seen"] = grouped["timestamp"].min()
    features["last_seen"] = grouped["timestamp"].max()
    features["active_days"] = (features["last_seen"] - features["first_seen"]).dt.days + 1
    features["txn_per_day"] = features["tx_count"] / features["active_days"]

    # Step 3: Merge with phishing labels
    labels = df[["from", "fromIsPhi"]].drop_duplicates(subset="from")
    dataset = features.merge(labels, left_index=True, right_on="from", how="left").fillna(0)
    dataset.rename(columns={"fromIsPhi": "label"}, inplace=True)
    dataset = dataset.sort_values("first_seen").reset_index(drop=True)

    sender_list = dataset["from"].values
    X = dataset.drop(columns=["from", "first_seen", "last_seen", "label"])
    y = dataset["label"]

    # Step 4: Train/Test Split
    split_point = int(0.8 * len(dataset))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    senders_test = sender_list[split_point:]

    # Step 5: Handle class imbalance with slight upsampling
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data["label"] == 0]
    minority = train_data[train_data["label"] == 1]
    
    # Upsample phishing class with more samples for better recall
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority) * 2, random_state=42)
    balanced_data = pd.concat([majority, minority_upsampled])
    X_train = balanced_data.drop("label", axis=1)
    y_train = balanced_data["label"]

    # Step 6: Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 7: Predict and Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Terminal output: Confusion Matrix and stats
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(cm)
    print("\nTrue Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)
    print("\nTotal Legitimate Accounts (label=0):", sum(y_test == 0))
    print("Total Phishing Accounts (label=1):", sum(y_test == 1))
    print("Predicted Phishing Accounts:", sum(y_pred == 1))
    print("Correctly Predicted Phishing Accounts (TP):", tp)

    # Show evaluation
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [acc, prec, rec, f1]
    })
    st.table(metrics_df)

    # Step 8: Identify phishing accounts
    results_df = X_test.copy()
    results_df["predicted_label"] = y_pred
    results_df["actual_label"] = y_test.values
    results_df["sender"] = senders_test
    phishing_accounts = results_df[results_df["predicted_label"] == 1]["sender"].unique()

    st.subheader("Detected Phishing Accounts")
    if len(phishing_accounts) > 0:
        st.write(f"{len(phishing_accounts)} accounts were flagged as phishing.")
        st.code("\n".join(phishing_accounts))
    else:
        st.warning("No phishing accounts detected.")

    # Step 9: Show Feature Importance
    st.subheader("Feature Importance (Logistic Regression)")
    feature_weights = model.coef_[0]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_weights, y=X.columns)
    plt.title("Importance of Each Feature")
    st.pyplot(plt)

    # Step 10: Comparison with Traditional Models
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame({
        "Model": ["Our Model (Logistic Regression)", "Node2Vec", "Trans2Vec"],
        "Accuracy": [round(acc, 2), 0.72, 0.81],
        "Precision": [round(prec, 2), 0.53, 0.61],
        "Recall": [round(rec, 2), 0.22, 0.31],
        "F1 Score": [round(f1, 2), 0.31, 0.41]
    })
    st.dataframe(comparison_df)

    # Step 11: Check a specific sender
    st.subheader("Check a Specific Sender Address")
    input_sender = st.text_input("Enter sender address:")

    if input_sender:
        if input_sender in dataset["from"].values:
            sender_row = dataset[dataset["from"] == input_sender].drop(columns=["from", "first_seen", "last_seen", "label"])
            sender_prediction = model.predict(sender_row)[0]
            if sender_prediction == 1:
                st.error("This address is predicted to be a phishing account.")
            else:
                st.success("This address is predicted to be a legitimate account.")
        else:
            st.warning("Sender address not found in the uploaded dataset.")
