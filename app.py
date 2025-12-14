# ============================================
# Kidney Disease Prediction Using Deep Learning
# Streamlit Cloud – FINAL REFINED VERSION
# ============================================

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

# --------------------------------------------
# Streamlit Page Config
# --------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Kidney Disease Prediction")
st.caption("Deep Learning Mini Project | Streamlit Cloud")

# --------------------------------------------
# Load & Clean Dataset (SAFE VERSION)
# --------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")

    # Replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop id column
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # Strip spaces in string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    # Encode target column
    df["classification"] = df["classification"].map({
        "ckd": 1,
        "notckd": 0
    })

    # Binary categorical mapping
    binary_map = {
        "yes": 1, "no": 0,
        "present": 1, "notpresent": 0,
        "abnormal": 1, "normal": 0,
        "poor": 1, "good": 0
    }
    df.replace(binary_map, inplace=True)

    # Fill missing values correctly
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

df = load_data()

# --------------------------------------------
# Split Data
# --------------------------------------------
X = df.drop("classification", axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------
# Build & Train ANN (Cached)
# --------------------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Handle class imbalance
    class_weights = {
        0: len(y_train) / (2 * np.sum(y_train == 0)),
        1: len(y_train) / (2 * np.sum(y_train == 1))
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

# --------------------------------------------
# Evaluate Model
# --------------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------
# User Input Section
# --------------------------------------------
st.subheader("📝 Enter Patient Medical Details")

user_input = []
for col in X.columns:
    value = st.number_input(
        label=col,
        value=float(X[col].median()),
        step=0.1
    )
    user_input.append(value)

# --------------------------------------------
# Prediction
# --------------------------------------------
if st.button("🔍 Predict Kidney Disease"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.error("⚠️ Result: **Chronic Kidney Disease Detected**")
    else:
        st.success("🎉 Result: **No Kidney Disease Detected**")

# --------------------------------------------
# Footer
# --------------------------------------------
st.markdown("---")
st.markdown(
    "**Neural Networks & Deep Learning Mini Project**  \n"
    "Technologies: Python, TensorFlow, Scikit-Learn, Streamlit"
)
