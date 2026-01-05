import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses **Machine Learning (Random Forest Algorithm)** to detect  
**malicious network traffic** in real time.

**Traffic Types**
- 🟢 Benign (Normal Traffic)
- 🔴 Malicious (DDoS / Attack)
""")

@st.cache_data
def load_data():
    csv_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)

        df.columns = df.columns.str.strip()

        required_columns = [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Packet Length Mean",
            "Active Mean",
            "Label"
        ]

        df = df[required_columns]

        df.columns = [
            "Destination_Port",
            "Flow_Duration",
            "Total_Fwd_Packets",
            "Packet_Length_Mean",
            "Active_Mean",
            "Label"
        ]

        df["Label"] = df["Label"].apply(
            lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
        )

        return df

    np.random.seed(42)
    n_samples = 5000

    data = {
        "Destination_Port": np.random.randint(1, 65535, n_samples),
        "Flow_Duration": np.random.randint(100, 100000, n_samples),
        "Total_Fwd_Packets": np.random.randint(1, 100, n_samples),
        "Packet_Length_Mean": np.random.uniform(10, 1500, n_samples),
        "Active_Mean": np.random.uniform(0, 1000, n_samples),
        "Label": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    return pd.DataFrame(data)


df = load_data()


st.sidebar.header("Control Panel")

train_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100)


X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - train_size) / 100, random_state=42
)

st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1️⃣ Model Training")

    if st.button("Train Model Now"):
        with st.spinner("Training Random Forest Classifier..."):
            model = RandomForestClassifier(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            st.session_state["model"] = model
        st.success("Model Training Completed!")

    if "model" in st.session_state:
        st.success("Model is Ready")


with col2:
    st.subheader("2️⃣ Performance Metrics")

    if "model" in st.session_state:
        model = st.session_state["model"]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc*100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Detected Attacks", int(np.sum(y_pred)))

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please train the model first.")

st.divider()
st.subheader("3️⃣ Live Traffic Simulator")

c1, c2, c3, c4 = st.columns(4)

flow_duration = c1.number_input("Flow Duration (ms)", 0, 100000, 500)
total_packets = c2.number_input("Total Forward Packets", 0, 500, 100)
packet_length = c3.number_input("Packet Length Mean", 0, 1500, 500)
active_mean = c4.number_input("Active Mean Time", 0, 1000, 50)

if st.button("Analyze Packet"):
    if "model" in st.session_state:
        model = st.session_state["model"]

       
        input_data = pd.DataFrame([{
            "Destination_Port": 80,
            "Flow_Duration": flow_duration,
            "Total_Fwd_Packets": total_packets,
            "Packet_Length_Mean": packet_length,
            "Active_Mean": active_mean
        }])

        pred = model.predict(input_data)

        if pred[0] == 1:
            st.error("🚨 ALERT: MALICIOUS TRAFFIC DETECTED!")
        else:
            st.success("✅ Traffic Status: BENIGN (Safe)")
    else:
        st.error("Please train the model first.")
