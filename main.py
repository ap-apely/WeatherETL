import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from weather_etl import etl_process
from logistic_model import LogisticRegression
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
from rich.panel import Panel
from rich.layout import Layout

console = Console()

def fetch_data_from_db():
    """
    Fetch data from SQLite database
    """
    conn = sqlite3.connect("data/weather.db")
    df = pd.read_sql("SELECT * FROM weather_data", conn)
    conn.close()
    return df

def one_hot_encode_season(df):
    """
    One-Hot Encoding для сезонов года
    """
    seasons = pd.get_dummies(df['season'], prefix='season', dtype=int)  # Используем dtype=int

    # Убедимся, что все сезоны присутствуют
    for season in ['season_spring', 'season_summer', 'season_autumn', 'season_winter']:
        if season not in seasons.columns:
            seasons[season] = 0
    return pd.concat([df, seasons], axis=1).drop(columns=['season'])

def standardize_features(X):
    """
    Стандартизация признаков
    
    Parameters:
    X (numpy.array): Array of features
    
    Returns:
    X (numpy.array): Standardized features
    mean (numpy.array): Mean of features
    std (numpy.array): Std of features
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1e-7, std)
    return (X - mean) / std, mean, std

def predict_custom_data(model, mean, std, features):
    """
    Make prediction for custom input data
    
    Parameters:
    model (LogisticRegression): Trained model
    mean (numpy.array): Mean of features
    std (numpy.array): Std of features
    features (dict): Custom input data with the following keys:
        temperature_2m_max (float): Max temperature
        temperature_2m_min (float): Min temperature
        surface_pressure_mean (float): Mean surface pressure
        windspeed_10m_max (float): Max wind speed
        season (str): Season
    
    Returns:
    prediction (int): Predicted class (0 or 1)
    probability (float): Predicted probability
    """
    # Create input array
    X = np.array([
        features['temperature_2m_max'],
        features['temperature_2m_min'],
        features['surface_pressure_mean'],
        features['windspeed_10m_max']
    ])
    
    # Standardize numerical features using saved mean and std
    X = (X - mean) / std
    
    # Add season one-hot encoding
    season = features['season'].lower()
    season_encoding = np.zeros(4)
    season_map = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
    if season in season_map:
        season_encoding[season_map[season]] = 1
    
    # Combine numerical and categorical features
    X = np.concatenate([X, season_encoding])
    
    # Reshape for prediction
    X = X.reshape(-1, 1)
    
    # Make prediction
    prediction = model.predict(X)
    probability = model.hypothesis(model.w, X, model.b)
    
    return prediction[0][0], probability[0][0]

def plot_training_history(costs, train_accuracies, test_accuracies):
    """
    Plot training metrics history
    """
    """
    Args:
    costs (list): Cost history
    train_accuracies (list): Train accuracy history
    test_accuracies (list): Test accuracy history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cost history
    iterations = range(0, len(costs) * 100, 100)
    ax1.plot(iterations, costs, 'b-')
    ax1.set_title('Cost History')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.grid(True)
    
    # Plot accuracy history
    ax2.plot(iterations, train_accuracies, 'g-', label='Train Accuracy')
    ax2.plot(iterations, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
def display_model_summary(model_params):
    """
    Display model parameters and metrics in a rich table
    """
    table = Table(title="Model Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Training Accuracy", f"{model_params['train_accuracy']:.2f}%")
    table.add_row("Testing Accuracy", f"{model_params['test_accuracy']:.2f}%")
    table.add_row("Final Cost", f"{model_params['costs'][-1]:.6f}")
    table.add_row("Bias", f"{model_params['b']:.6f}")
    table.add_row("Number of Features", str(model_params['w'].shape[0]))
    
    console.print(Panel(table, title="[bold blue]Model Performance", border_style="blue"))
def visualize_prediction(features, prediction, probability):
    """
    Visualize the prediction with matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Temperature and Pressure with improved styling
    ax1.plot([features['temperature_2m_min'], features['temperature_2m_max']], 
             [0, 1], 'b-', label='Temperature Range', linewidth=2)
    ax1.axhline(y=features['surface_pressure_mean']/1000, 
                color='r', linestyle='-', label='Pressure (kPa)', linewidth=2)
    ax1.set_title('Temperature and Pressure', fontsize=12, pad=15)
    ax1.set_xlabel('Temperature (°C)', fontsize=10)
    ax1.set_ylabel('Normalized Scale', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Prediction Probability with improved styling
    """
    Create a pie chart with matplotlib
    """
    colors = ['#FF9999', '#90EE90']  # Light red and light green
    labels = ['No Rain', 'Rain']
    sizes = [1 - probability, probability]
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 10})
    ax2.set_title(f'Precipitation Prediction\nPredicted: {"Rain" if prediction == 1 else "No Rain"}',
                  fontsize=12, pad=15)
    
    plt.tight_layout()
    plt.show()
"""
Train a logistic regression model on the weather dataset
and make predictions with custom weather data
"""

if __name__ == "__main__":
    with console.status("[bold green]Running ETL process...") as status:
        etl_process()
        status.update("[bold green]Loading and preparing data...")
        
        df = fetch_data_from_db()
        df = one_hot_encode_season(df)

    features = ['temperature_2m_max', 'temperature_2m_min', 'surface_pressure_mean', 
                'windspeed_10m_max', 'season_spring', 'season_summer', 'season_autumn', 'season_winter']
    
    X = df[features].values.astype('float32')
    Y = df['precipitation'].values.astype('float32')

    rprint(Panel.fit(f"[cyan]Initial shapes - X: {X.shape} Y: {Y.shape}"))

    numerical_features = ['temperature_2m_max', 'temperature_2m_min', 'surface_pressure_mean', 'windspeed_10m_max']
    numerical_indices = [features.index(f) for f in numerical_features]
    
    with console.status("[bold yellow]Preprocessing data..."):
        X_numerical = X[:, numerical_indices]
        X_numerical, mean, std = standardize_features(X_numerical)
        X[:, numerical_indices] = X_numerical

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.reshape(1, -1)
        y_test = y_test.reshape(1, -1)

    rprint(Panel.fit(
        f"[cyan]Final shapes:\nX_train: {X_train.shape} y_train: {y_train.shape}\n" +
        f"X_test: {X_test.shape} y_test: {y_test.shape}"
    ))

    with console.status("[bold blue]Training model...") as status:
        model = LogisticRegression(learning_rate=0.0001, num_iterations=1000000)
        d = model.train_model(X_train=X_train, Y_train=y_train, 
                            X_test=X_test, Y_test=y_test, print_cost=True)
    
    display_model_summary(d)
    
    plot_training_history(d['costs'], d['train_accuracies'], d['test_accuracies'])

    while True:
        if not console.input("\n[bold cyan]Would you like to make a prediction? (y/n/q): ").lower() in ['y', 'yes']:
            break
            
        console.print("\n[bold green]Enter weather data:[/bold green]")
        try:
            custom_features = {
                'temperature_2m_max': float(console.input("[cyan]Maximum temperature (°C): ")),
                'temperature_2m_min': float(console.input("[cyan]Minimum temperature (°C): ")),
                'surface_pressure_mean': float(console.input("[cyan]Mean surface pressure (hPa): ")),
                'windspeed_10m_max': float(console.input("[cyan]Maximum wind speed (km/h): ")),
                'season': console.input("[cyan]Season (spring/summer/autumn/winter): ")
            }
            
            prediction, probability = predict_custom_data(model, mean, std, custom_features)
            
            result_color = "green" if prediction == 1 else "red"
            console.print(Panel(f"[bold {result_color}]Prediction: {'Rain' if prediction == 1 else 'No Rain'}"))
            console.print(Panel(f"[bold blue]Probability of rain: {probability:.2%}"))
            
            visualize_prediction(custom_features, prediction, probability)
            
        except ValueError as e:
            console.print("[bold red]Error: Please enter valid numeric values[/bold red]")
