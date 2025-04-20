import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

# Set Streamlit Page Config with custom theme
st.set_page_config(page_title="Solar Power Prediction", layout="wide")

# Custom CSS for Navigation Bar in Header
st.markdown("""
    <style>
        .header {
            background-color: #2b2b2b;
            padding: 15px;
            text-align: center;
            color: #32cd32;
            font-size: 24px;
            font-weight: bold;
        }
        .nav-links {
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }
        .nav-item {
            margin: 0 20px;
            font-size: 18px;
            color: #ffffff;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Prediction"])

# Load Dataset
df = pd.read_csv("D:\\Solarpowerprediction\\Data\\processed_data.csv")
df = df.dropna()

# Split data into features (X) and target (y)
features = df.drop(columns=['generated_power_kw'])
target = df['generated_power_kw']

# Home Page
if menu == "Home":
        # Title and Subtitle
    st.markdown("""
        <h1 style='text-align: center; font-weight: bold;'>Solar Power Prediction System</h1>
        <h3 style='text-align: center;'>AI in Green Technology</h3>
    """, unsafe_allow_html=True)

    # Overview Section
    st.header("📌 Overview")
    st.write("Artificial Intelligence (AI) in green technology plays a pivotal role in optimizing renewable energy sources. It enables accurate forecasting, efficient energy distribution, and enhanced sustainability efforts. By leveraging AI-driven machine learning models, we can predict solar power generation, minimize energy wastage, and improve grid stability. This project integrates AI to create a scalable and reliable solar power prediction system, aiding in the global transition to sustainable energy solutions.")

    # Display Images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("D:\\Solarpowerprediction\\assets\\solar.png", caption="Solar Panels & AI", use_column_width=True)
    with col2:
        st.image("D:\\Solarpowerprediction\\assets\\energyeficiency.png", caption="Smart Grid Optimization", use_column_width=True)
    with col3:
        st.image("D:\\Solarpowerprediction\\assets\\ai.png", caption="AI in Energy Forecasting", use_column_width=True)

    # Problem Statement
    st.header("📌 Problem Statement")
    col1,col2 = st.columns(2)
    with col1:
        st.write("\n\n🌍 **Challenges in Solar Energy Utilization**")
        st.write("- **Unpredictable Power Generation**: Weather variability affects solar output.")
        st.write("- **Energy Waste & Inefficiencies**: Excess power generated is often underutilized.")
        st.write("- **Grid Instability**: Without accurate predictions, grids struggle with demand-supply balance.")

    with col2:
        st.image("D:\\Solarpowerprediction\\assets\\keychalenges.png", use_column_width=True)

    # Sustainable Technology Goals
    st.header("📌 Sustainable Technology Goals")
    st.write("✅ **Aligning with UN Sustainable Development Goals (SDGs)**")
    st.write("- **Goal 7**: Affordable & Clean Energy 🌱")
    st.write("- **Goal 9**: Industry Innovation & Infrastructure 🚀")
    st.write("- **Goal 13**: Climate Action 🌍")

    # Solution
    st.header("📌 Solution")
    col1,col2 = st.columns(2)
    with col1:
        st.write("✅ AI-Powered Solar Power Prediction")
        st.write("- Uses **Machine Learning (Linear Regression & Random Forest)** to forecast solar energy output.")
        st.write("- Real-time data visualization through an interactive Streamlit dashboard.")
        st.write("- Enhances energy grid planning & sustainable power distribution.")
    with col2:
        st.image("D:\\Solarpowerprediction\\assets\\solution.png", use_column_width=True)
        


    # Tools & Technologies
    st.header("📌 Tools & Technologies")
    st.image("D:\\Solarpowerprediction\\assets\\tools&tech.png",  use_column_width=True)

    # Methodology
    st.header("📌 Methodology")
    methodology_data = pd.DataFrame({
        "Step": ["Data Collection", "Feature Engineering", "Model Training", "Evaluation & Optimization", "Deployment"],
        "Process": ["Gathered weather & solar energy data.", "Created new predictive variables (Cloud Cover, Wind Speed, Sun Position).",
                    "Trained Linear Regression & Random Forest models.", "Compared RMSE & R² scores, tuned hyperparameters.", "Built a Streamlit-based UI for predictions."],
        "Outcome": ["Structured dataset for ML models.", "Improved model accuracy.", "Optimized solar power forecasting.", "Enhanced performance & accuracy.", "User-friendly, real-time forecasting."]
    })
    st.table(methodology_data)
    st.image("D:\\Solarpowerprediction\\assets\\process.png", use_column_width=True)

    # Approaches
    st.header("📌 Approaches")
    approaches_data = pd.DataFrame({
        "Approach": ["Linear Regression", "Random Forest", "Feature Engineering", "Real-Time AI Dashboard"],
        "Why Chosen?": ["Simple, interpretable, works well for linear patterns.", "Handles non-linearity & complex relationships.",
                        "Extracts new weather-based features.", "Streamlit-based live forecasting."],
        "Impact": ["Baseline model for comparison.", "Higher accuracy & robustness.", "Enhances model performance.", "Industry-ready for smart grids."]
    })
    st.table(approaches_data)
    

    
    # Conclusion
    st.header("📌 Conclusion")
    st.write("✅ Developed an AI-powered solar prediction system that improves energy planning.")
    st.write("✅ Real-time interactive dashboard enables accurate forecasting & smart grid integration.")
    st.write("✅ Optimized model performance through feature engineering & hyperparameter tuning.")
    st.write("✅ Scalable & Industry-Ready for sustainability & renewable energy adoption.")

    # Related Links
    st.header("📌 Related Links")
    st.write("🔗[Visit the source code of this project](https://github.com/monasri001/SolarPowerPrediction-Edunet-Internship)")
    st.write("🔗 [UN Sustainable Development Goals](https://sdgs.un.org/goals)")
    st.write("🔗 [AI in Renewable Energy Research](https://www.nrel.gov/)")
    st.write("🔗 [Streamlit for AI Applications](https://streamlit.io/)")

    # References
    st.header("📌 References")
    st.write("- 📄 **R. M. T. Ramirez, K. A. Dey, and J. A. Smith, Machine Learning for Accelerating the Discovery of High-Performance Low-Cost Solar Cells. Available: https://arxiv.org/abs/2212.13893**")
    st.write("- 📑 **Mehmet Yesilbudak; Medine Çolak; Ramazan Bayindir, A review of data mining and solar power prediction. Available:https://ieeexplore.ieee.org/document/7884507**")
    st.write("- 📑 ** Pooja Rani; Rashmi Taya; V Padmanabha Reddy, A Review on Solar Energy and Different Electricity Generations. Available: https://ieeexplore.ieee.org/document/10451552**")

# Data Exploration
elif menu == "Data Exploration":
    st.title("📊 Data Exploration")
    st.title("Dataset Overview")

    # General Information
    st.subheader("General Information")
    st.markdown("""
    - **Number of Records:** 4,213  
    - **Number of Features:** 21  
    - **Data Types:**  
    - 17 columns → **float64**  
    - 4 columns → **int64**  
    - **Target Variable:** `generated_power_kw` (Represents the power generated in kilowatts)  
    """)

    # Key Features
    st.subheader("Key Features")

    st.markdown("### **Weather Conditions:**")
    st.markdown("""
    - Temperature: `temperature_2_m_above_gnd`  
    - Humidity: `relative_humidity_2_m_above_gnd`  
    - Pressure: `mean_sea_level_pressure_MSL`  
    - Precipitation: `total_precipitation_sfc`, `snowfall_amount_sfc`  
    - Cloud Cover:  
    - `total_cloud_cover_sfc`  
    - `high_cloud_cover_high_cld_lay`  
    - `medium_cloud_cover_mid_cld_lay`  
    - `low_cloud_cover_low_cld_lay`  
    - Radiation: `shortwave_radiation_backwards_sfc`  
    """)

    st.markdown("### **Wind Parameters:**")
    st.markdown("""
    - **Wind Speed & Direction:**  
    - `wind_speed_10_m_above_gnd`, `wind_direction_10_m_above_gnd`  
    - `wind_speed_80_m_above_gnd`, `wind_direction_80_m_above_gnd`  
    - `wind_speed_900_mb`, `wind_direction_900_mb`  
    - `wind_gust_10_m_above_gnd`  
    """)

    st.markdown("### **Solar Angles:**")
    st.markdown("""
    - `angle_of_incidence`  
    - `zenith`  
    - `azimuth`  
    """)

    st.markdown("### **Power Output:**")
    st.markdown("- `generated_power_kw` (**Target Variable**)")

    # Initial Observations
    st.subheader("Initial Observations")
    st.markdown("""
    - The dataset appears **clean** with no missing values.  
    - It includes **weather, wind, and solar position data**, which likely influence power generation.  
    - The dataset may be useful for **modeling and predicting solar power generation** based on environmental conditions.  
    """)
    st.title("Dataset Preview")
    st.dataframe(df.head())
    st.title("Statistical summaries")
    st.write(df.describe())
    
    st.title("Visualization")
    # Plot distribution of power
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Generated Power (kW)")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['generated_power_kw'], bins=30, kde=True)
        plt.title('Distribution of Generated Power (kW)')
        plt.xlabel('Generated Power (kW)')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    with col2:
        st.markdown(
        """
        <div style="display: flex; height: 100%; align-items: center; justify-content: left;">
            <div>
                <h3>Power Generation Insights</h3>
                <p>The generated power (kW) follows a <b>right-skewed distribution</b>, with most values on the lower side and occasional high peaks. This indicates fluctuating solar power generation based on varying conditions.</p>
                <p><b>Key Observations:</b></p>
                <ul>
                    <li><b>Mean Power Output:</b> 1,134.35 kW</li>
                    <li><b>Median Power Output:</b> 971.64 kW</li>
                    <li><b>Generated Power Range:</b> 0.0006 kW to 3,056.79 kW</li>
                    <li><b>High Variability:</b> Standard deviation of 937.96 kW</li>
                    <li><b>Right-Skewed:</b> More low-power occurrences with occasional high peaks</li>
                    <li><b>Influencing Factors:</b> Likely dependent on weather, cloud cover, and solar angles</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Add a thick bolded line
    st.markdown("<hr style='border: 4px solid white;'>", unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")
    col3,col4 = st.columns(2)
    with col3:
        st.markdown(
        """
        <div style="display: flex; height: 100%; align-items: center; justify-content: left;">
            <div>
                <h3>Features Relationship Insights</h3>
                <p>The correlation heatmap visualizes the relationships between numerical features
                in the dataset. Each cell shows the Pearson correlation coefficient between
                two variables, where:</p>
                <ul>
                    <li><b>+1</b>: Strong positive correlation</li>
                    <li><b>-1</b>: Strong negative correlation</li>
                    <li><b>0</b>: No linear correlation</li>
                </ul>
                <p>The color intensity reflects the strength of the correlation,
                with warm colors indicating positive correlations and cool colors indicating negative correlations.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
        
    with col4:
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
    # Add a thick bolded line
    st.markdown("<hr style='border: 4px solid white;'>", unsafe_allow_html=True)

    st.subheader("Temperature vs Wind speed vs Power generated")
    col5,col6 = st.columns(2)
    with col5:
        # 3D Scatter Plot temperature vs humidity vs generated power
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df["temperature_2_m_above_gnd"], df["wind_speed_10_m_above_gnd"], df["generated_power_kw"])
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Wind Speed")
        ax.set_zlabel("Generated Power")
        plt.title("Temperature vs Wind speed vs Power generated")
        st.pyplot(plt)
    
    with col6:
        st.markdown(
        """
        <div style="display: flex; height: 100%; align-items: center; justify-content: left;">
            <div>
                <h3>Impact of Temperature and Wind Speed on Power Generation</h3>
                <p>The plot visualizes how <b>temperature</b> and <b>wind speed</b> impact solar power generation.</p>
                <p><b>Key Observations:</b></p>
                <ul>
                    <li><b>Higher temperatures</b> generally correlate with higher power generation.</li>
                    <li><b>Wind speed</b> has a less clear impact, though moderate wind speeds may aid cooling, improving efficiency.</li>
                    <li><b>Clusters at lower power values</b> suggest that some conditions are not optimal for solar generation.</li>
                    <li><b>Some outliers</b> indicate occasional high power output, likely influenced by ideal sunlight conditions.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
        )


    # with col6: 
    # Add a thick bolded line
    st.markdown("<hr style='border: 4px solid white;'>", unsafe_allow_html=True)

    #Pie chart distribution for cloud,wind,power analysis
    st.subheader("Distribution for Cloud,Wind,Generated Power analysis")
    # Define categories
    categories = {
        "Cloud Cover Category": ("total_cloud_cover_sfc", [0, 25, 50, 75, 100], ["Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast"]),
        "Wind Speed Category": ("wind_speed_10_m_above_gnd", [0, 5, 15, 30, 100], ["Calm", "Breeze", "Windy", "Storm"]),
        "Power Category": ("generated_power_kw", [0, 500, 1500, 3000, df["generated_power_kw"].max()], ["Low", "Medium", "High", "Very High"])
    }

    # Apply categorization
    for key, (col, bins, labels) in categories.items():
        df[key] = pd.cut(df[col], bins=bins, labels=labels)

    # Plot pie charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, _) in zip(axes, categories.items()):
        df[key].value_counts().plot.pie(autopct="%1.1f%%", ax=ax, colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
        ax.set_title(key)

    plt.tight_layout()
    st.pyplot(plt)

    # Add a thick bolded line
    st.markdown("<hr style='border: 4px solid white;'>", unsafe_allow_html=True)

    

# Model Training
elif menu == "Model Training":
    
    st.title("📈 Model Training")
    import streamlit as st

    # Create two columns for Linear Regression and Random Forest
    col6, col7 = st.columns(2)

    # Define box style using markdown with CSS
    box_style = """
        <div style="
            border: 3px solid white; 
            padding: 15px; 
            border-radius: 10px; 
            background-color: #262730; 
            text-align: left;
        ">
            <h3 style="color: white; margin-bottom: 5px;">{title}</h3>
            <p style="color: white;">{content}</p>
        </div>
    """

    # Add content inside styled boxes
    with col6:
        st.markdown(box_style.format(
            title="Linear Regression",
            content="""
            <ul>
                <li>A simple and interpretable machine learning model for regression.</li>
                <li>Assumes a linear relationship between input features and the target variable.</li>
                <li>Minimizes the difference between predicted and actual values using a straight-line equation.</li>
                <li>Good for understanding how features impact the target variable.</li>
            </ul>
            """
        ), unsafe_allow_html=True)

    with col7:
        st.markdown(box_style.format(
            title="Random Forest",
            content="""
            <ul>
                <li>A powerful ensemble learning method that combines multiple decision trees.</li>
                <li>Reduces overfitting by averaging multiple tree predictions.</li>
                <li>Works well for both regression and classification tasks.</li>
                <li>Can handle large datasets and complex patterns better than linear models.</li>
            </ul>
            """
        ), unsafe_allow_html=True)


    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)


    joblib.dump(lr_model, "linear_regression.pkl")
    joblib.dump(rf_model, "random_forest.pkl")

    st.title("Model performance comparison based on Metrics")
    # Create three columns
    col8, col9, col10 = st.columns(3)

    # Define box style using markdown with CSS
    box_style = """
        <div style="
            border: 3px solid white; 
            padding: 15px; 
            border-radius: 10px; 
            background-color: #262730; 
            text-align: left;
        ">
            <h3 style="color: white; margin-bottom: 5px;">{title}</h3>
            <p style="color: white;">{content}</p>
        </div>
    """

    
    # Add content inside styled boxes
    with col9:
        st.markdown(box_style.format(
            title="R² (R-squared)",
            content="""
            <ul>
                <li>A higher R² score is generally good as it indicates how much variance in the dependent variable is explained by the model.</li>
                <li><b>R² = 1</b> → Perfect fit (explains all variance).</li>
                <li><b>R² close to 1</b> → Good fit (explains most variance).</li>
                <li><b>R² close to 0</b> → Poor fit (explains little variance).</li>
                <li><b>R² < 0</b> → Worse than a simple mean prediction (bad model).</li>
            </ul>
            """
        ), unsafe_allow_html=True)

    with col8:
        st.markdown(box_style.format(
            title="RMSE (Root Mean Squared Error)",
            content="""
            <ul>
                <li>Measures the average prediction error.</li>
                <li><b>Lower RMSE is better</b> (smaller errors).</li>
                <li><b>Higher RMSE is worse</b> (larger deviations from actual values).</li>
                <li><b>RMSE is in the same units as the target variable</b>, making it easy to interpret.</li>
                <li><b>High R² + Low RMSE = Good model</b></li>
                <li><b>Low R² + High RMSE = Poor model</b></li>
                <li><b>High R² + High RMSE = Possible overfitting.</b></li>
            </ul>
            """
        ), unsafe_allow_html=True)

    with col10:
        st.markdown(box_style.format(
            title="MAE (Mean Absolute Error)",
            content="""
            <ul>
                <li>Measures the average absolute difference between predicted and actual values.</li>
                <li><b>Lower MAE is better</b> (smaller errors).</li>
                <li><b>Higher MAE is worse</b> (larger errors).</li>
                <li><b>MAE is in the same units as the target variable</b> for easy interpretation.</li>
                <li>Unlike RMSE, <b>MAE does not heavily penalize large errors</b> (no squaring).</li>
            </ul>
            """
        ), unsafe_allow_html=True)


        
        # Sample Data
    data = {
        'Model': ['Linear Regression', 'Random Forest'],
        'Root Mean Squared Error (RMSE)': [f"{lr_rmse:.4f}", f"{rf_rmse:.4f}"],
        'R_squared Score': [f"{lr_r2:.4f}", f"{rf_r2:.4f}"],
        'Mean Absolute Error (MAE)': [f"{lr_mae:.4f}", f"{rf_mae:.4f}"]
    }

    df1 = pd.DataFrame(data)

    # Apply custom table styling using markdown and CSS
    st.markdown("""
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                border: 3px solid white;
                padding: 12px;
                text-align: center;
                color: white;
                font-size: 16px;
            }
            th {
                background-color: #444;
                font-size: 18px;
            }
            tr:nth-child(even) {
                background-color: #333;
            }
            tr:nth-child(odd) {
                background-color: #262730;
            }
        </style>
    """, unsafe_allow_html=True)
    # Display Styled Table
    st.table(df1)

    st.subheader("Model Performance Comparison")
    models = ["Linear Regression", "Random Forest"]
    rmse_scores = [lr_rmse, rf_rmse]
    r2_scores = [lr_r2, rf_r2]
    mae_scores =[lr_mae,rf_mae]

    plt.figure(figsize=(20, 5))
    plt.subplot(1,3, 1)
    sns.barplot(x=models, y=rmse_scores)
    plt.title("RMSE Comparison")

    plt.subplot(1,3, 2)
    sns.barplot(x=models, y=r2_scores)
    plt.title("R² Score Comparison")

    plt.subplot(1,3,3)
    sns.barplot(x=models, y=mae_scores)
    plt.title("MAE Score Comparison")
    st.pyplot(plt)


# Prediction Page
# Prediction
elif menu == "Prediction":
    st.title("🔮 Solar Power Prediction")

    # Load Trained Model
    model_choice = st.radio("Select Model", ["Linear Regression", "Random Forest"])
    model = joblib.load(f"{'linear_regression' if model_choice == 'Linear Regression' else 'random_forest'}.pkl")

    # Input Features
    st.subheader("Enter Weather Conditions")
    input_data = {}
    for col in features.columns:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

    # Predict Button
    if st.button("Predict Power Output"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"Predicted Power Output: {prediction[0]:.2f} kW")
