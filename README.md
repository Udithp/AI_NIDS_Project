
# AI-Based Network Intrusion Detection System (NIDS)

## 📌 Project Overview
The AI-Based Network Intrusion Detection System (NIDS) is a machine learning–driven cybersecurity application designed to detect malicious network activities in real time. The system classifies network traffic as **benign** or **malicious** using a Random Forest algorithm and provides an interactive web-based dashboard for monitoring and analysis.

---

## 🎯 Objectives
- Detect network intrusions such as DDoS attacks using machine learning  
- Improve accuracy over traditional rule-based intrusion detection systems  
- Provide real-time traffic analysis through an interactive dashboard  
- Support both simulated and real-world network traffic data  

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Machine Learning Algorithm:** Random Forest  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Web Framework:** Streamlit  
- **Visualization:** Matplotlib, Seaborn  
- **Dataset:** CIC-IDS2017 (Friday Afternoon DDoS CSV)  
- **IDE:** Visual Studio Code  

---

## 📂 Project Structure
```

AI_NIDS_Project/
│
├── nids_main.py
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (optional)
├── README.md

````

---

## 📊 Data Management Strategy
- **Simulation Mode (Default):**  
  Uses synthetic network traffic generated with NumPy for immediate demonstration.
  
- **Production Mode (Optional):**  
  Supports real-world network traffic data from the CIC-IDS2017 dataset.  
  If the CSV file is present in the project folder, the system automatically loads it.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher  
- Visual Studio Code (recommended)

### Install Dependencies
```bash
pip install pandas numpy scikit-learn streamlit seaborn matplotlib
````

---

## ▶️ How to Run the Project

1. Navigate to the project directory:

   ```bash
   cd AI_NIDS_Project
   ```
2. Run the application:

   ```bash
   streamlit run nids_main.py
   ```
3. Open your browser and go to:

   ```
   http://localhost:8501
   ```

---

## 🖥️ Application Features

* Interactive Streamlit dashboard
* Train Random Forest model with a single click
* View accuracy and confusion matrix
* Live traffic simulator for real-time intrusion detection
* Automatic handling of feature consistency

---

## 🛠️ Troubleshooting

* **Streamlit not recognized:**
  Run `python -m streamlit run nids_main.py`
* **Module not found error:**
  Ensure all dependencies are installed correctly
* **Browser does not open:**
  Manually visit `http://localhost:8501`

---

## 📌 Conclusion

This project demonstrates how artificial intelligence and machine learning can be applied to modern cybersecurity challenges. The AI-Based Network Intrusion Detection System provides an effective, scalable, and academic-ready solution for detecting network anomalies and malicious activities.

---

## 📚 References

* Python Documentation
* Scikit-learn Documentation
* Streamlit Documentation
* CIC-IDS2017 Dataset – Canadian Institute for Cybersecurity
