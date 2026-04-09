import os
import random
import requests
from datetime import timedelta

import numpy as np
import pandas as pd


def load_historical_sales(data_path='data/store_sale.csv'):
    """Load the Kaggle sales dataset and aggregate to monthly total sales."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Group by month and sum
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('year_month')['sales'].sum().reset_index()
    monthly['date'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly[['date', 'sales']].reset_index(drop=True)

    monthly['sales_diff'] = monthly['sales'].diff()
    monthly = monthly.dropna().reset_index(drop=True)
    return monthly


def build_sales_supervised(monthly_df, n_lags=12):
    """Build lag features for the forecasting model."""
    supervised = monthly_df[['sales_diff']].copy()
    for i in range(1, n_lags + 1):
        supervised[f'month_{i}'] = supervised['sales_diff'].shift(i)
    supervised = supervised.dropna().reset_index(drop=True)
    return supervised


def simulate_inventory_data(dates, supplier_list=None, seed=42):
    """Simulate inventory levels, supplier names, and days-to-stockout for Layer 1."""
    random.seed(seed)
    np.random.seed(seed)

    if supplier_list is None:
        supplier_list = ['Supplier-A', 'Supplier-B', 'Supplier-C', 'Supplier-D']

    rows = []
    for idx, date in enumerate(dates):
        supplier = random.choice(supplier_list)
        base_demand = max(1, int(20000 + 5000 * np.sin(idx / 3.0)))
        inventory = float(np.clip(base_demand * random.uniform(0.25, 1.5), 500, 120000))
        days_to_stockout = float(np.clip(inventory / max(base_demand / 30.0, 1.0), 1.0, 45.0))

        rows.append({
            'date': date,
            'supplier_name': supplier,
            'current_inventory': inventory,
            'days_to_stockout': days_to_stockout
        })

    return pd.DataFrame(rows)


def compute_weather_disruption_score(weather_texts):
    """Convert weather descriptions to a 0-1 disruption score."""
    scores = []
    for text in weather_texts:
        text_lower = str(text).lower()
        score = 0.0
        if any(word in text_lower for word in ['storm', 'cyclone', 'flood', 'torrential', 'heatwave', 'hail']):
            score += 0.6
        if any(word in text_lower for word in ['rain', 'extreme', 'wind', 'fog', 'thunder', 'lightning']):
            score += 0.3
        if any(word in text_lower for word in ['clear', 'sunny', 'dry', 'mild']):
            score -= 0.2
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return scores


def fetch_weatherapi_history(city, date, api_key):
    """Fetch historical weather for a city from WeatherAPI.com."""
    endpoint = 'http://api.weatherapi.com/v1/history.json'
    params = {
        'key': api_key,
        'q': city,
        'dt': date.strftime('%Y-%m-%d')
    }
    response = requests.get(endpoint, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    day = data.get('forecast', {}).get('forecastday', [])[0].get('day', {})
    return {
        'condition': day.get('condition', {}).get('text', ''),
        'avgtemp_c': day.get('avgtemp_c', 0.0),
        'maxwind_kph': day.get('maxwind_kph', 0.0),
        'totalprecip_mm': day.get('totalprecip_mm', 0.0),
        'daily_chance_of_rain': day.get('daily_chance_of_rain', 0),
        'daily_chance_of_snow': day.get('daily_chance_of_snow', 0)
    }


def score_weather_observation(observation):
    """Convert real weather data into a disruption score between 0 and 1."""
    condition = str(observation.get('condition', '')).lower()
    score = 0.0

    if observation.get('totalprecip_mm', 0.0) >= 15.0:
        score += 0.35
    elif observation.get('totalprecip_mm', 0.0) >= 5.0:
        score += 0.20

    if observation.get('maxwind_kph', 0.0) >= 60.0:
        score += 0.25
    elif observation.get('maxwind_kph', 0.0) >= 35.0:
        score += 0.12

    if any(term in condition for term in ['storm', 'cyclone', 'flood', 'torrential', 'hail', 'severe thunderstorm']):
        score += 0.30
    elif any(term in condition for term in ['rain', 'thunder', 'shower', 'squall']):
        score += 0.15
    elif any(term in condition for term in ['clear', 'sunny', 'fine', 'dry']):
        score -= 0.20

    return float(np.clip(score, 0.0, 1.0))


def get_weather_signals_for_cities(city_list, start_date, end_date, weather_api_key=None):
    """Build weather disruption signals for the selected cities."""
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    rows = []
    for city in city_list:
        for date in dates:
            if weather_api_key:
                try:
                    observation = fetch_weatherapi_history(city, date, weather_api_key)
                    weather_text = observation['condition']
                    disruption_score = score_weather_observation(observation)
                except Exception:
                    weather_text = 'API error fallback'
                    disruption_score = float(np.clip(random.random() * 0.4, 0.0, 1.0))
            else:
                weather_text = random.choice([
                    'Heavy rain and wind',
                    'Light showers',
                    'Clear sky',
                    'Cloudy with moderate rain',
                    'Torrential flood warning',
                    'Sunny and mild'
                ])
                disruption_score = compute_weather_disruption_score([weather_text])[0]

            rows.append({
                'city': city,
                'date': date,
                'weather_description': weather_text,
                'weather_disruption_score': disruption_score
            })
    return pd.DataFrame(rows)


def score_news_disruption(headlines):
    """Score NewsData.io headlines into a disruption signal between 0 and 1."""
    scores = []
    for headline in headlines:
        text = str(headline).lower()
        score = 0.0
        if any(term in text for term in ['strike', 'flood', 'protest', 'election', 'lockdown', 'transport', 'delay', 'blockade', 'rail', 'road', 'political', 'policy', 'festival']):
            score += 0.25
        if any(term in text for term in ['crisis', 'flood', 'strike', 'shut', 'failure', 'collapse']):
            score += 0.30
        if any(term in text for term in ['good news', 'restored', 'reopened', 'normal', 'smooth']):
            score -= 0.15
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return scores


def fetch_newsdata_io(api_key, query='india disruption', from_date=None, to_date=None, page_size=50):
    """Fetch India-focused news headlines from NewsData.io."""
    endpoint = 'https://newsdata.io/api/1/news'
    params = {
        'apikey': api_key,
        'q': query,
        'language': 'en',
        'country': 'in',
        'page': 0,
        'page_size': page_size
    }
    if from_date is not None:
        params['from_date'] = from_date.strftime('%Y-%m-%d')
    if to_date is not None:
        params['to_date'] = to_date.strftime('%Y-%m-%d')

    headlines = []
    while True:
        response = requests.get(endpoint, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        for article in data.get('results', []):
            headlines.append(article.get('title') or article.get('description') or '')

        if data.get('nextPage'):
            params['page'] += 1
        else:
            break

    return headlines


def build_layer1_dataset(
    sales_path='store_sale.csv',
    news_api_key=None,
    weather_api_key=None,
    supplier_list=None,
    city_list=None,
    weather_start=None,
    weather_end=None,
    news_query='india disruption'
):
    """Collect and merge Layer 1 data sources."""
    monthly_sales = load_historical_sales(sales_path)
    supervised_sales = build_sales_supervised(monthly_sales)

    inventory_data = simulate_inventory_data(monthly_sales['date'], supplier_list=supplier_list)

    if city_list is None:
        city_list = ['Mumbai', 'Bengaluru', 'Delhi', 'Kolkata']
    if weather_start is None:
        weather_start = monthly_sales['date'].min()
    if weather_end is None:
        weather_end = monthly_sales['date'].max()

    weather_signals = get_weather_signals_for_cities(city_list, weather_start, weather_end, weather_api_key=weather_api_key)
    weather_summary = (weather_signals.groupby('date', as_index=False)
                       ['weather_disruption_score']
                       .mean()
                       .rename(columns={'weather_disruption_score': 'weather_disruption_score'}))

    if news_api_key:
        headlines = fetch_newsdata_io(news_api_key, query=news_query, from_date=weather_start, to_date=weather_end)
    else:
        headlines = [
            'Regional transport strike causes shipment delay',
            'Heavy flood warnings in coastal cities',
            'Election-related road closures in multiple districts',
            'Local temple festival increases delivery demand',
            'Normal traffic and stable supplier operations'
        ]

    news_scores = score_news_disruption(headlines)
    disruption_signal = float(np.clip(np.mean(news_scores), 0.0, 1.0))

    layer1_df = monthly_sales[['date', 'sales', 'sales_diff']].copy()
    layer1_df = layer1_df.merge(inventory_data[['date', 'current_inventory', 'days_to_stockout']], on='date', how='left')
    layer1_df = layer1_df.merge(weather_summary, on='date', how='left')
    layer1_df['disruption_signal'] = disruption_signal
    layer1_df['supplier_risk_score'] = np.clip(
        0.2 + 0.6 * (1.0 - layer1_df['current_inventory'] / layer1_df['sales'].replace(0, 1)),
        0.0, 1.0
    )

    return {
        'monthly_sales': monthly_sales,
        'supervised_sales': supervised_sales,
        'inventory_data': inventory_data,
        'weather_signals': weather_signals,
        'news_headlines': headlines,
        'layer1_df': layer1_df
    }
