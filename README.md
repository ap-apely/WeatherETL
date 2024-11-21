# 🌦️ WeatherETL: Weather Prediction with Machine Learning

**WeatherETL** is a sophisticated ETL (Extract, Transform, Load) and machine learning project that predicts precipitation based on weather data. Using logistic regression and real-time weather data from Open-Meteo API, it provides accurate rainfall predictions with beautiful visualizations.

---

## 📜 Table of Contents

1. [🌟 Features](#-features)
2. [📁 Project Structure](#-project-structure)
3. [🚀 How to Run](#-how-to-run)
4. [📊 Data Pipeline](#-data-pipeline)
5. [🤖 Machine Learning Model](#-machine-learning-model)
6. [📈 Visualization](#-visualization)
7. [🧰 Technologies Used](#-technologies-used)

---

## 🌟 Features

- **Automated ETL Pipeline**: Fetches and processes weather data from Open-Meteo API
- **Custom Logistic Regression**: Implementation with numerical stability and advanced optimization
- **Interactive Predictions**: Real-time weather prediction with user input
- **Rich Visualizations**: Training metrics and prediction probability visualization
- **Data Persistence**: SQLite database integration for weather data storage
- **Beautiful CLI**: Rich console output with progress tracking and styled tables

---

## 📁 Project Structure

```
├───data
│   └───weather.db        # SQLite database for weather data
├───main.py              # Main application entry point
├───logistic_model.py    # Custom logistic regression implementation
└───weather_etl.py       # ETL pipeline implementation
```

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/WeatherETL.git
   cd WeatherETL
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

---

## 📊 Data Pipeline

### Data Collection
- Fetches weather data from Open-Meteo API
- Parameters include:
  - Maximum/Minimum Temperature
  - Surface Pressure
  - Wind Speed
  - Precipitation

### Data Processing
- Season determination based on date
- One-hot encoding for seasonal data
- Feature standardization
- Precipitation binary classification

### Data Storage
- SQLite database with weather_data table
- Structured schema for efficient data retrieval

---

## 🤖 Machine Learning Model

### Logistic Regression Features
- Custom implementation with gradient descent
- Numerical stability improvements:
  - Sigmoid function clipping
  - Small epsilon for log calculations
  - Standardized feature scaling
- Configurable hyperparameters:
  - Learning rate
  - Number of iterations
  - Random weight initialization

### Model Performance Tracking
- Cost history visualization
- Training/Testing accuracy monitoring
- Real-time training progress display

---

## 📈 Visualization

### Training Metrics
- Cost convergence plot
- Accuracy progression charts
- Model performance summary table

### Prediction Visualization
- Temperature and pressure relationship plot
- Rain probability pie chart
- Interactive prediction interface

---

## 🧰 Technologies Used

- **Data Processing**: NumPy, Pandas
- **API Integration**: Requests
- **Database**: SQLite
- **Visualization**: Matplotlib
- **UI Enhancement**: Rich
- **Machine Learning**: Custom Implementation
- **Data Validation**: Scikit-learn (train_test_split)

---

## 📝 Configuration

The model can be configured through various parameters:

```python
model = LogisticRegression(
    learning_rate=0.0001,
    num_iterations=1000000
)
```

### API Configuration
```python
PARAMS = {
    "latitude": 55.75,
    "longitude": 37.61,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "surface_pressure_mean",
        "windspeed_10m_max"
    ],
    "timezone": "Europe/Moscow",
    "past_days": 90
}
```

---

**WeatherETL** – Your comprehensive solution for weather data analysis and precipitation prediction! 🌦️
