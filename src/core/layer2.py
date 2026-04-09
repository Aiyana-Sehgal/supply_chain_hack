"""
Layer 2: Feature Engineering & State Preparation

Takes raw Layer 1 outputs and prepares consolidated feature sets
for downstream forecasting and RL decision-making.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_features(df, columns, method='minmax', feature_range=(0, 1)):
    """
    Normalize specified columns using MinMax or Standard scaling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Column names to normalize
    method : str
        'minmax' or 'standard'
    feature_range : tuple
        (min, max) for MinMaxScaler
    
    Returns:
    --------
    pd.DataFrame, scaler object
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("method must be 'minmax' or 'standard'")
    
    # Only scale non-null values
    valid_mask = df_copy[columns].notna().all(axis=1)
    valid_data = df_copy.loc[valid_mask, columns]
    
    # Only fit/transform if there are valid rows
    if len(valid_data) > 0:
        df_copy.loc[valid_mask, columns] = scaler.fit_transform(valid_data)
    
    return df_copy, scaler


def handle_outliers(series, method='iqr', threshold=1.5):
    """
    Detect and clip outliers in a series.
    
    Parameters:
    -----------
    series : pd.Series
        Input series
    method : str
        'iqr' (interquartile range) or 'zscore'
    threshold : float
        IQR multiplier (1.5 is standard) or Z-score threshold
    
    Returns:
    --------
    pd.Series with outliers clipped
    """
    series = series.copy()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        series = series.clip(lower=lower_bound, upper=upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        series = series[z_scores <= threshold]
    
    return series


def aggregate_weather_by_window(weather_signals_df, method='mean', window_days=30):
    """
    Aggregate per-city weather disruption scores to a single regional signal.
    
    Parameters:
    -----------
    weather_signals_df : pd.DataFrame
        Output from Layer 1's get_weather_signals_for_cities()
        Columns: city, date, weather_description, weather_disruption_score
    method : str
        Aggregation: 'mean', 'max', 'weighted'
    window_days : int
        Aggregation window (days)
    
    Returns:
    --------
    pd.DataFrame with regional weather disruption time series
    """
    df = weather_signals_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if method == 'mean':
        agg_df = df.groupby('date', as_index=False)['weather_disruption_score'].mean()
        agg_df.columns = ['date', 'regional_weather_disruption']
    
    elif method == 'max':
        agg_df = df.groupby('date', as_index=False)['weather_disruption_score'].max()
        agg_df.columns = ['date', 'regional_weather_disruption']
    
    elif method == 'weighted':
        # Weight by city population (simplified: equal weight here)
        agg_df = df.groupby('date', as_index=False)['weather_disruption_score'].mean()
        agg_df.columns = ['date', 'regional_weather_disruption']
    
    return agg_df


def compute_rolling_statistics(df, column, window=3):
    """
    Compute rolling mean, std, and min/max for a column over a window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    column : str
        Column name to compute rolling stats
    window : int
        Rolling window size (in periods)
    
    Returns:
    --------
    pd.DataFrame with rolling statistics appended
    """
    df = df.copy()
    df[f'{column}_rolling_mean'] = df[column].rolling(window=window, min_periods=1).mean()
    df[f'{column}_rolling_std'] = df[column].rolling(window=window, min_periods=1).std()
    df[f'{column}_rolling_min'] = df[column].rolling(window=window, min_periods=1).min()
    df[f'{column}_rolling_max'] = df[column].rolling(window=window, min_periods=1).max()
    return df


def aggregate_suppliers(inventory_data, method='risk_weighted'):
    """
    Aggregate supplier-level inventory into a consolidated supply chain state.
    
    Parameters:
    -----------
    inventory_data : pd.DataFrame
        Output from Layer 1's simulate_inventory_data()
        Columns: date, supplier_name, current_inventory, days_to_stockout
    method : str
        'mean', 'min', 'risk_weighted'
    
    Returns:
    --------
    pd.DataFrame with aggregated supplier metrics per date
    """
    df = inventory_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if method == 'mean':
        agg = df.groupby('date', as_index=False).agg({
            'current_inventory': 'mean',
            'days_to_stockout': 'mean'
        })
        agg.columns = ['date', 'avg_inventory', 'avg_days_to_stockout']
    
    elif method == 'min':
        agg = df.groupby('date', as_index=False).agg({
            'current_inventory': 'min',
            'days_to_stockout': 'min'
        })
        agg.columns = ['date', 'min_inventory', 'min_days_to_stockout']
    
    elif method == 'risk_weighted':
        # Lower days_to_stockout = higher risk = higher weight
        df['risk_weight'] = 1.0 / (df['days_to_stockout'] + 1)
        df['risk_weight'] = df['risk_weight'] / df.groupby('date')['risk_weight'].transform('sum')
        
        # Group by date and aggregate
        agg_list = []
        for date_val in df['date'].unique():
            subset = df[df['date'] == date_val]
            weighted_inv = (subset['current_inventory'] * subset['risk_weight']).sum()
            weighted_days = (subset['days_to_stockout'] * subset['risk_weight']).sum()
            agg_list.append({'date': date_val, 'weighted_inventory': weighted_inv, 'weighted_days_to_stockout': weighted_days})
        
        agg = pd.DataFrame(agg_list)
    
    return agg


def create_composite_risk_signal(layer1_df, inventory_agg, weather_agg, 
                                 weights=None):
    """
    Combine supplier risk, disruption, and operational signals into a composite risk score.
    
    Parameters:
    -----------
    layer1_df : pd.DataFrame
        Layer 1 consolidated output
    inventory_agg : pd.DataFrame
        Aggregated inventory metrics
    weather_agg : pd.DataFrame
        Aggregated weather metrics
    weights : dict
        Weights for different risk components
        Default: {'supplier': 0.3, 'disruption': 0.4, 'weather': 0.3}
    
    Returns:
    --------
    pd.DataFrame with composite_risk_score column
    """
    if weights is None:
        weights = {'supplier': 0.3, 'disruption': 0.4, 'weather': 0.3}
    
    df = layer1_df[['date', 'supplier_risk_score', 'disruption_signal']].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Merge aggregated metrics
    df = df.merge(inventory_agg, on='date', how='left')
    df = df.merge(weather_agg, on='date', how='left')
    
    # Normalize components to [0, 1]
    supplier_risk = df['supplier_risk_score'].fillna(0.5)
    disruption_risk = df['disruption_signal'].fillna(0.3)
    weather_risk = df['regional_weather_disruption'].fillna(0.2)
    
    # Composite score (weighted average)
    df['composite_risk_score'] = (
        weights['supplier'] * supplier_risk +
        weights['disruption'] * disruption_risk +
        weights['weather'] * weather_risk
    )
    
    df['composite_risk_score'] = df['composite_risk_score'].clip(0, 1)
    
    return df[['date', 'composite_risk_score']]


def build_rl_state_features(layer1_df, weather_agg, inventory_agg, 
                            forecast_df=None, lookback_window=3):
    """
    Prepare consolidated state features for the RL agent.
    
    Parameters:
    -----------
    layer1_df : pd.DataFrame
        Layer 1 consolidated output
    weather_agg : pd.DataFrame
        Aggregated weather metrics
    inventory_agg : pd.DataFrame
        Aggregated inventory metrics
    forecast_df : pd.DataFrame, optional
        Demand forecast from Layer 3 (columns: month, predicted_sales)
    lookback_window : int
        Number of historical periods to include in features
    
    Returns:
    --------
    pd.DataFrame with state representation for RL agent
    """
    state_df = layer1_df[['date', 'sales', 'sales_diff', 'current_inventory', 
                          'days_to_stockout', 'supplier_risk_score', 
                          'disruption_signal']].copy()
    
    # Merge aggregated signals
    state_df = state_df.merge(weather_agg, on='date', how='left')
    state_df = state_df.merge(inventory_agg, on='date', how='left')
    
    # Add demand forecast if available
    if forecast_df is not None:
        forecast_df = forecast_df.copy()
        forecast_df.columns = ['date', 'predicted_demand', 'forecast_diff']
        state_df = state_df.merge(forecast_df, on='date', how='left')
    
    # Compute rolling features (volatility, trends)
    state_df['sales_volatility'] = state_df['sales'].rolling(window=lookback_window).std()
    state_df['demand_trend'] = state_df['sales'].rolling(window=lookback_window).mean()
    state_df['inventory_trend'] = state_df['current_inventory'].rolling(window=lookback_window).mean()
    
    # Fill NaNs from rolling
    state_df = state_df.fillna(method='bfill').fillna(method='ffill')
    
    return state_df


def build_layer2_dataset(layer1_bundle, inventory_method='risk_weighted',
                         weather_method='mean', include_rolling=True):
    """
    Main Layer 2 pipeline: consolidate Layer 1 outputs into ML-ready features.
    
    Parameters:
    -----------
    layer1_bundle : dict
        Output from Layer 1's build_layer1_dataset()
        Keys: 'monthly_sales', 'inventory_data', 'weather_signals', 
              'news_headlines', 'layer1_df'
    inventory_method : str
        Aggregation method for supplier inventory
    weather_method : str
        Aggregation method for weather signals
    include_rolling : bool
        Whether to include rolling statistics
    
    Returns:
    --------
    dict with consolidated Layer 2 outputs
    """
    # Extract Layer 1 outputs
    layer1_df = layer1_bundle['layer1_df'].copy()
    inventory_data = layer1_bundle['inventory_data'].copy()
    weather_signals = layer1_bundle['weather_signals'].copy()
    
    # Ensure datetime columns
    layer1_df['date'] = pd.to_datetime(layer1_df['date'])
    inventory_data['date'] = pd.to_datetime(inventory_data['date'])
    weather_signals['date'] = pd.to_datetime(weather_signals['date'])
    
    # Aggregate inventory across suppliers
    inventory_agg = aggregate_suppliers(inventory_data, method=inventory_method)
    
    # Aggregate weather across cities
    weather_agg = aggregate_weather_by_window(weather_signals, method=weather_method)
    
    # Create composite risk signal
    risk_df = create_composite_risk_signal(layer1_df, inventory_agg, weather_agg)
    
    # Merge all signals into a consolidated dataframe
    consolidated_df = layer1_df[['date', 'sales', 'sales_diff', 'current_inventory',
                                  'days_to_stockout', 'supplier_risk_score', 
                                  'disruption_signal']].copy()
    consolidated_df = consolidated_df.merge(inventory_agg, on='date', how='left')
    consolidated_df = consolidated_df.merge(weather_agg, on='date', how='left')
    consolidated_df = consolidated_df.merge(risk_df, on='date', how='left')
    
    # Add rolling statistics if requested
    if include_rolling:
        consolidated_df = compute_rolling_statistics(consolidated_df, 'sales', window=3)
        consolidated_df = compute_rolling_statistics(consolidated_df, 'current_inventory', window=3)
        consolidated_df = compute_rolling_statistics(consolidated_df, 'composite_risk_score', window=3)
    
    # Normalize key features (exclude date column)
    features_to_normalize = [
        'sales', 'current_inventory', 'supplier_risk_score', 
        'disruption_signal', 'regional_weather_disruption', 'composite_risk_score'
    ]
    existing_features = [f for f in features_to_normalize if f in consolidated_df.columns]
    
    # Only normalize if we have valid features
    scaler = None
    if len(existing_features) > 0:
        consolidated_df, scaler = normalize_features(consolidated_df, existing_features, method='minmax')
    else:
        existing_features = []
    
    return {
        'consolidated_df': consolidated_df,
        'inventory_agg': inventory_agg,
        'weather_agg': weather_agg,
        'risk_df': risk_df,
        'scaler': scaler,
        'feature_columns': existing_features
    }
