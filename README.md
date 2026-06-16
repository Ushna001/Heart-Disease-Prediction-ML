# ❤️ Heart Disease Prediction ML System

An end-to-end Machine Learning application that predicts the risk of heart disease based on patient health parameters. The project combines data preprocessing, machine learning models, and an interactive Streamlit web application to provide prediction results with probability analysis and automated PDF reports.

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. This project aims to use Machine Learning techniques to assist in early risk prediction by analyzing important clinical features such as age, blood pressure, cholesterol level, chest pain type, and other health indicators.

The system allows users to enter patient information and receive:
- Heart disease risk prediction
- Prediction probability score
- Health recommendations
- Downloadable PDF prediction report

> **Note:** This application is developed for educational purposes and should not be considered a replacement for professional medical diagnosis.

---

## 🚀 Features

✅ Machine Learning-based heart disease prediction  
✅ Data preprocessing and feature analysis  
✅ Model training and evaluation  
✅ Probability-based risk assessment  
✅ Interactive web interface using Streamlit  
✅ Automated PDF report generation  
✅ User-friendly prediction workflow  

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Pandas | Data processing and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning model development |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| FPDF | PDF report generation |

---

## 📂 Project Structure

```
Heart-Disease-Prediction-ML/
│
├── app.py                      # Streamlit application
├── model/                      # Trained ML models
├── dataset/                    # Dataset files
├── requirements.txt             # Required libraries
├── README.md                    # Project documentation
└── reports/                     # Generated PDF reports
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ushna001/Heart-Disease-Prediction-ML.git
```

### 2. Navigate to project directory

```bash
cd Heart-Disease-Prediction-ML
```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧠 Machine Learning Workflow

The project follows these steps:

1. Data Collection  
2. Data Cleaning & Preprocessing  
3. Feature Selection  
4. Model Training  
5. Model Evaluation  
6. Prediction Generation  
7. User Interface Deployment  

---

## 📊 Input Parameters

The model uses patient health attributes such as:

- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Other clinical indicators

---

## 📈 Application Screenshots

(Add screenshots of your Streamlit application here)

Example:

```
![Home Page](images/home.png)
![Prediction Result](images/result.png)
```

---

## 📄 Report Generation

The application automatically generates a PDF report containing:

- Patient input details
- Prediction result
- Risk probability
- Clinical disclaimer
- Recommendations

---

## 🎯 Future Improvements

- Improve prediction accuracy using advanced ML algorithms
- Add model comparison dashboard
- Deploy application on cloud platforms
- Integrate explainable AI (SHAP/LIME)
- Add user authentication and database storage

---

## 👨‍💻 Author

**Ushna**

GitHub:  
https://github.com/Ushna001

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub.
