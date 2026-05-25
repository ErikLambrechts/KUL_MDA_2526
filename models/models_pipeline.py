import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Import all 4 algorithms
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_all_district_models_by_year(df):
    print("\n--- Step 7: Data Cleansing & Chronological Training (Train: 2024, Test: 2025) ---")
    
    # 1. DEFINE FEATURES
    weather_features = ['temp_max', 'temp_min', 'temp_avg', 'precip_quantity', 'humidity_relative', 'pressure', 'sun_duration', 'short_wave_from_sky', 'evapotrans_ref'] 
    available_weather = [feat for feat in weather_features if feat in df.columns]
    
    # 2. RIGOROUS DATA CLEANSING
    required_cols = ['amount', 'date', 'district'] + available_weather
    df_ml = df.dropna(subset=required_cols).copy()
    
    # Poisson regression mathematically requires target values >= 0
    df_ml = df_ml[df_ml['amount'] >= 0]
    
    # Ensure date column is explicitly datetime
    df_ml['date'] = pd.to_datetime(df_ml['date'])

    # 3. FEATURE ENGINEERING FROM DATES
    df_ml['month'] = df_ml['date'].dt.month.astype(str)
    df_ml['day_of_week'] = df_ml['date'].dt.day_name()
    df_ml['is_weekend'] = df_ml['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    categorical_features = ['month', 'day_of_week']
    numeric_features = available_weather + ['is_weekend']
    
    # 4. PREPROCESSING PIPELINE
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    
    # Define our suite of algorithms
    models_to_train = {
        'Linear_Regression': LinearRegression(),
        'Poisson_Regression': PoissonRegressor(max_iter=300),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    all_district_results = {}
    unique_districts = df_ml['district'].unique()
    
    # 5. LOOP PER DISTRICT
    for district in unique_districts:
        district_data = df_ml[df_ml['district'] == district]
        
        # [NEW] CHRONOLOGICAL SPLIT FOR THIS DISTRICT
        # Filter training data exclusively to 2024
        train_data = district_data[(district_data['date'] >= '2024-01-01') & (district_data['date'] <= '2024-12-31')]
        
        # Filter testing data exclusively to 2025
        test_data = district_data[(district_data['date'] >= '2025-01-01') & (district_data['date'] <= '2025-12-31')]
        
        # Skip districts if they are missing data for either of the tracking windows
        if len(train_data) < 30 or len(test_data) < 10:
            print(f" Skipping [{district}]: Insufficient data split (Train rows: {len(train_data)}, Test rows: {len(test_data)}).")
            continue
            
        print(f"\nTraining for District: {district} (2024 Train: {len(train_data)} rows | 2025 Test: {len(test_data)} rows)...")
        
        # Separate features and targets
        X_train = train_data[categorical_features + numeric_features]
        y_train = train_data['amount']
        
        X_test = test_data[categorical_features + numeric_features]
        y_test = test_data['amount']
        
        district_summary = {}
        
        # Train and evaluate each model type
        for model_name, model_algorithm in models_to_train.items():
            # Bundle preprocessing and algorithm into an end-to-end pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model_algorithm)
            ])
            
            try:
                # Train using ONLY 2024 records
                pipeline.fit(X_train, y_train)
                
                # Predict and validate using ONLY 2025 records
                y_pred = pipeline.predict(X_test)
                
                # Evaluate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                district_summary[model_name] = {
                    'pipeline': pipeline,
                    'r2_score': r2,
                    'rmse': rmse
                }
                print(f"   -> {model_name:18} | 2025 R² Score: {r2:6.3f} | 2025 RMSE: {rmse:.1f}")
                
            except Exception as e:
                print(f"   -> [ERROR] Failed to train {model_name}: {e}")
                
        all_district_results[district] = district_summary
        
    return all_district_results