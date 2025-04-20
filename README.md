# 🌞 Solar Power Output Prediction Using Linear Regression

[![Website](https://img.shields.io/badge/Live%20App-Visit-blue?style=for-the-badge)](https://solarpowerprediction-en3y.onrender.com)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-blue?style=for-the-badge&logo=github)](https://github.com/monasri001/Solar_Power_Output_Prediction_Using_Linear_Regression)

## 📌 Project Title
**Solar Power Prediction Using Linear Regression**

## 🧠 Problem Statement

The goal of this project is to predict **solar power generation** based on **historical weather data** using **Linear Regression**. Accurately forecasting solar energy output is essential for optimizing **energy distribution**, especially in regions that rely heavily on solar energy. By leveraging environmental features like **temperature, humidity, and solar irradiance**, we aim to build an effective predictive model.

---

## 🌱 Sustainable Green Technology Goal

This project promotes **sustainable energy sources** by:
- Forecasting solar power output
- Optimizing solar panel deployment
- Reducing dependency on non-renewables
- Enhancing energy management efficiency

---

## 🚀 How We Achieve It

1. **📊 Data Collection** – Historical data including temperature, humidity, radiation, etc.
2. **🧹 Data Preprocessing** – Handling missing values, normalization, and feature scaling.
3. **📈 Exploratory Data Analysis** – Visualizing trends and feature relationships.
4. **🔧 Model Building** – Training a Linear Regression model for prediction.
5. **✅ Evaluation** – Using RMSE and R-squared metrics for accuracy.
6. **🌐 Web Deployment** – Frontend and backend deployment using Flask & Render.

---

## 📅 Project Timeline & Progress

### ✅ **Week 1: Data Preprocessing**
- 🔹 Imported necessary libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- 🔹 Loaded and accessed dataset: Features include temperature, humidity, irradiance, etc.
- 🔹 Cleaned data: Handled missing values, duplicates, and outliers
- 🔹 Data statistics: Used `.describe()` and `.info()` to get insights
- 🔹 Standardization & normalization of features

---

### ✅ **Week 2: Exploratory Data Analysis (EDA)**

#### 📌 **Univariate Analysis**
- Histograms & KDE: Normal distributions (temperature), right-skewed (humidity, wind)
- Box Plots: Identified outliers
- Pie Charts: Proportions of variables

#### 📌 **Bivariate Analysis**
- Scatter Plots: Radiation ↗ Power, Cloud Cover ↘ Power
- Correlation Heatmap:
  - 🔺 Positive: Shortwave Radiation (0.9), Temperature (0.6)
  - 🔻 Negative: Cloud Cover (-0.7), Humidity (-0.6)

#### 📌 **Multivariate Analysis**
- Pair Plot: Identified key relationships
- 3D Scatter Plot: Highlighted dependency of power on radiation
- Contour Plot: Visualizing zenith, azimuth, and power

---

### ✅ **Week 3: Model Building & Training**

- 🔹 Chose **Linear Regression** for simplicity and interpretability
- 🔹 Splitted dataset into train/test (80/20)
- 🔹 Trained the model on normalized data
- 🔹 Evaluated with:
  - **RMSE (Root Mean Square Error)**
  - **R-squared**
- 🔹 Interpreted coefficients to understand feature impact

---

### ✅ **Week 4: Web App Deployment**

- 🔹 Built a clean and interactive **Flask** web interface
- 🔹 User inputs: Temperature, humidity, radiation, etc.
- 🔹 Prediction results displayed with clean UI
- 🔹 Deployed to **Render**:  
  🔗 [Live App Link](https://solarpowerprediction-en3y.onrender.com)

---

## 🖥️ How to Run the Project Locally

1. Clone the Repository
    ```bash
    git clone https://github.com/monasri001/Solar_Power_Output_Prediction_Using_Linear_Regression.git
    cd Solar_Power_Output_Prediction_Using_Linear_Regression

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Open the VS Code.
4. Navigate to the solarapp.py file and open it.
   ```bash
   streamlit run solarapp.py
5. Run the cells in the notebook.
