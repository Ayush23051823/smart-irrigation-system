# 🌱 Smart Irrigation System

An AI-powered irrigation management system that optimizes water usage using Machine Learning and real-time environmental inputs.

---

## 📌 Overview

This project predicts whether irrigation is required and estimates the optimal water quantity based on soil and weather conditions. It integrates a Machine Learning backend with an interactive Streamlit dashboard for real-time decision-making.

> 📍 This project is developed under the guidance of **Ankit Raj**

---

## ⚙️ Features

- 🌾 Irrigation Need Prediction (Classification)
- 💧 Water Requirement Estimation (Regression)
- 📊 Interactive Dashboard (Streamlit UI)
- 📈 Real-time Graph Visualization
- 📥 Downloadable Reports (CSV)
- 🧠 AI-based decision support system

---

## 🧪 Tech Stack

| Category        | Technology |
|----------------|-----------|
| Language       | Python |
| ML Models      | Random Forest (Classifier & Regressor) |
| Libraries      | Pandas, NumPy, Scikit-learn |
| Visualization  | Streamlit |
| Deployment     | Local / Streamlit |

---

## 🧠 Machine Learning Pipeline

1. Data preprocessing & cleaning  
2. Feature encoding (One-hot encoding)  
3. Model training:
   - Classification → Irrigation Decision  
   - Regression → Water Amount  
4. Model evaluation:
   - Accuracy Score  
   - Mean Absolute Error (MAE)  
5. Integration with UI  

---

## 📊 Outputs

- Irrigation decision (Yes / No)
- Recommended water (in mm)
- Soil moisture trend analysis
- Comparative visualization graphs

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/smart-irrigation-system.git

# Navigate to project
cd smart-irrigation-system

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
📁 Project Structure
smart-irrigation-system/
│
├── app.py                 # Streamlit UI
├── main.py                # ML training script
├── model.pkl             # Classification model
├── water_model.pkl       # Regression model
├── irrigation_prediction.csv
├── README.md
└── requirements.txt
📸 Screenshots (Optional but recommended)

Add your UI screenshots here for better presentation

📈 Future Improvements
🌐 Cloud deployment (Streamlit Cloud / AWS)
📡 IoT sensor integration
📊 Advanced analytics dashboard
🤖 Deep Learning models
👨‍💻 Authors
Ayush Kumar
Aparajita Shrivastava
Sanjolee Singh
Bansika Binoi
📜 License

This project is for academic and research purposes.

⭐ Acknowledgment

Special thanks to Mr.Ankit Raj Sir for guidance and support.


---

