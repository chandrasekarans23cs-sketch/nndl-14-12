# ----------------------------------------
# Kidney Disease Prediction Using Deep Learning
# Streamlit Single File Application
# ----------------------------------------

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
# Streamlit Page Config
# ----------------------------------------
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Kidney Disease Prediction Using Deep Learning")
st.write("Predict whether a patient has **Chronic Kidney Disease (CKD)** using a Deep Neural Network.")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")
    return df

df = load_data()

# ----------------------------------------
# Data Preprocessing
# ----------------------------------------
df.replace('?', np.nan, inplace=True)

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# Build ANN Model
# ----------------------------------------
@st.cache_resource
def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    return model

model = build_model()

# ----------------------------------------
# Model Accuracy
# ----------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ----------------------------------------
# User Input Section
# ----------------------------------------
st.header("📝 Enter Patient Medical Details")

input_data = []

for col in X.columns:
    value = st.number_input(f"{col}", value=float(X[col].mean()))
    input_data.append(value)

# ----------------------------------------
# Prediction
# ----------------------------------------
if st.button("🔍 Predict Kidney Disease"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    if prediction > 0.5:
        st.error("⚠️ Prediction Result: Patient has **Chronic Kidney Disease (CKD)**")
    else:
        st.success("🎉 Prediction Result: Patient does **NOT** have Kidney Disease")

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.markdown("### 📌 Mini Project – Neural Networks & Deep Learning")
st.markdown("Developed using **Python, TensorFlow & Streamlit**")
