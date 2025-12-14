# ========================================
# Kidney Disease Prediction Using Deep Learning
# Refined Streamlit App (Cloud Safe)
# ========================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress TF logs

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Kidney Disease Prediction")
st.caption("Deep Learning (ANN) | Streamlit Cloud Deployment")

# ----------------------------------------
# Load & Preprocess Data (Cached)
# ----------------------------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("kidney_disease.csv")

    df.replace("?", np.nan, inplace=True)

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    X = df.drop("classification", axis=1)
    y = df["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, X_train, X_test, y_train, y_test, scaler

X, X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

# ----------------------------------------
# Build & Train Model (Cached)
# ----------------------------------------
@st.cache_resource
def train_model(X_train, y_train, input_dim):
    model = Sequential([
        Dense(32, activation="relu", input_dim=input_dim),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,          # Reduced for Streamlit Cloud stability
        batch_size=16,
        verbose=0
    )

    return model

model = train_model(X_train, y_train, X_train.shape[1])

# ----------------------------------------
# Model Performance
# ----------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ----------------------------------------
# User Input Section
# ----------------------------------------
st.subheader("📝 Enter Patient Details")

input_values = []
for col in X.columns:
    value = st.number_input(
        label=col,
        value=float(X[col].mean()),
        step=0.1
    )
    input_values.append(value)

# ----------------------------------------
# Prediction
# ----------------------------------------
if st.button("🔍 Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.error("⚠️ **Result: Chronic Kidney Disease Detected**")
    else:
        st.success("🎉 **Result: No Kidney Disease Detected**")

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.markdown(
    "**Mini Project – Neural Networks & Deep Learning**  \n"
    "Built with **TensorFlow, Scikit-learn & Streamlit**"
)
