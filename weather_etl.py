import sqlite3
import requests
import pandas as pd
from datetime import datetime

API_URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": 55.75,
    "longitude": 37.61,
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,surface_pressure_mean,windspeed_10m_max",
    "timezone": "Europe/Moscow",
    "past_days": 90  # Get 90 days of historical data
}


def initialize_database():
    """
    Initializes the SQLite database by creating the weather_data table
    if it does not exist.
    """
    conn = sqlite3.connect("data/weather.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            date TEXT PRIMARY KEY,
            temperature_2m_max REAL,
            temperature_2m_min REAL,
            surface_pressure_mean REAL,
            windspeed_10m_max REAL,
            season TEXT,
            precipitation INTEGER
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")


def determine_season(date):
    """
    Determines the season for a given date.

    Args:
        date (datetime.date): Date to determine season for.

    Returns:
        str: Season name (winter, spring, summer, or autumn).
    """
    month = date.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

def extract_weather_data():
    """
    Fetches weather data from Open-Meteo API and returns it as a Pandas DataFrame.
    """
    response = requests.get(API_URL, params=PARAMS)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data["daily"])
    else:
        raise Exception("Failed to fetch data from API.")

def transform_weather_data(df):
    """
    Transform weather data to match the table structure.
    """
    df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    df['precipitation'] = (df['precipitation_sum'] > 0).astype(int) 
    df['season'] = pd.to_datetime(df['time']).apply(determine_season)
    return df[['date', 'temperature_2m_max', 'temperature_2m_min', 'surface_pressure_mean', 
               'windspeed_10m_max', 'season', 'precipitation']]

def load_to_db(df):
    """
    Load data to database, replacing existing data.
    """
    conn = sqlite3.connect("data/weather.db")
    df.to_sql("weather_data", conn, if_exists="replace", index=False)
    conn.close()
    print("Data loaded to database.")

def etl_process():
    """
    Execute the ETL process to fetch, transform and load data to DB
    """
    print("Starting ETL process...")
    initialize_database()
    data = extract_weather_data()
    transformed_data = transform_weather_data(data)
    load_to_db(transformed_data)
    print("ETL completed successfully!")
