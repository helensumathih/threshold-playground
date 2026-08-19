import streamlit as st
import pandas as pd
import joblib

st.title("Spam Threshold Playground")
with st.expander("What does this app show?"):
    st.write("""
            This app trains a spam classifier and lets you adjust the **decision threshold** - the cutoff probability
            above which a message is labelled "spam". 
            
            - **Higher threshold** -> fewer false alarms (higer precision), but more spam slips through (lower recall) 
            - **Lower threshold** -> catches more spam (higher recall), but more real messages get wrongly flagged (lower precision)

            There is no threshold that maximizes both - this app lets you explore that tradeoff directly.
            """)

st.write("Drag the slider to see the classification threshold affects precision and recall.")

#Load model and vectorizer (once)
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

#Load test data again we we have labelled examples to evaluate against
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
df["target"] = (df["label"] == "spam").astype(int)


#Reuse the SAME train/test split logic as explore.py so results are consistent
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"])

X_test_vec = vectorizer.transform(X_test)
probs = model.predict_proba(X_test_vec)[:, 1]

# ---THE SLIDER---
threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)

#Try it yourself
st.subheader("Try it yourself")
user_input = st.text_area("Type a message to classify: ", "Congratulations! You won a free prize, click here to claim.")
if user_input:
    user_vec = vectorizer.transform([user_input])
    user_probs = model.predict_proba(user_vec)[:, 1][0]
    label = "SPAM" if user_probs >= threshold else "HAM"
    st.write(f"**Prediction:** {label} (spam probability: {user_probs:.3f}, threshold: {threshold:.2f})")

predictions = (probs >= threshold).astype(int)

TP = ((predictions == 1) & (y_test == 1)).sum()
FP = ((predictions == 1) & (y_test == 0)).sum()
FN = ((predictions == 0) & (y_test == 1)).sum()
TN = ((predictions == 0) & (y_test == 0)).sum()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

col1, col2 = st.columns(2)
col1.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")

st.subheader("Confusion Matrix")
matrix = pd.DataFrame([[TP, FN], [FP, TN]], index=["Actually Spam", "Actually Ham"], columns=["Predicted Spam", "Predicted Ham"])

st.table(matrix)