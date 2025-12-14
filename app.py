# ========================================
# Kidney Disease Prediction Using Deep Learning
# High Accuracy Streamlit App
# ========================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(page_title="Kidney Disease Prediction", page_icon="🧠")
st.title("🧠 Kidney Disease Prediction (Deep Learning)")

# ----------------------------------------
# Load & Clean Data
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")
    df.replace("?", np.nan, inplace=True)

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # Target mapping
    df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

    # Binary mappings
    binary_map = {
        "yes": 1, "no": 0,
        "present": 1, "notpresent": 0,
        "abnormal": 1, "normal": 0,
        "poor": 1, "good": 0
    }

    for col in df.columns:
        df[col] = df[col].replace(binary_map)

    # Fill missing values
    for col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df

df = load_data()

# ----------------------------------------
# Train-Test Split
# ----------------------------------------
X = df.drop("classification", axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# Build Model
# ----------------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    class_weights = {
        0: len(y_train) / (2 * sum(y_train == 0)),
        1: len(y_train) / (2 * sum(y_train == 1))
    }

    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        verbose=0,
        class_weight=class_weights
    )

    return model

model = train_model(X_train, y_train)

# ----------------------------------------
# Evaluate
# ----------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ----------------------------------------
# User Input
# ----------------------------------------
st.subheader("📝 Patient Details")
input_data = []

for col in X.columns:
    value = st.number_input(col, float(X[col].median()))
    input_data.append(value)

# ----------------------------------------
# Prediction
# ----------------------------------------
if st.button("🔍 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    result = model.predict(input_scaled)[0][0]

    if result > 0.5:
        st.error("⚠️ Chronic Kidney Disease Detected")
    else:
        st.success("🎉 No Kidney Disease Detected")
