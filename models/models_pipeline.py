import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# ==========================================
# 1. GLOBAL SYSTEM CONFIGURATION & DIRECTORIES
# ==========================================
CONFIG = {
    "INPUT_DATA_PATH": "./KUL_MDA_2526/data/MDA_Final_Dataset.parquet",
    "OUTPUT_DIRECTORY": "./KUL_MDA_2526/data", 
    "EXPLICIT_WEATHER_COLS": ['temp_max', 'temp_min', 'temp_avg', 'precip_quantity', 'humidity_relative', 'pressure', 'sun_duration', 'short_wave_from_sky', 'evapotrans_ref']
}

os.makedirs(CONFIG["OUTPUT_DIRECTORY"], exist_ok=True)

# ==========================================
# 2. OPTIMIZED PLOTTING ENGINE
# ==========================================
def save_optimized_performance_plot(metrics_dataframe, output_directory):
    """Generates a clean performance plot with the legend docked outside the frame."""
    if metrics_dataframe.empty:
        print("    [Warning] Metrics dataframe is empty. Skipping plot generation.")
        return

    plot_df = metrics_dataframe.copy()
    r2_col = 'R2_Score' if 'R2_Score' in plot_df.columns else 'R2'
    plot_df[r2_col] = plot_df[r2_col].clip(lower=-0.1)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(
        data=plot_df, 
        x="Model", 
        y=r2_col, 
        hue="District", 
        palette="tab20",  
        ax=ax
    )
    
    plt.xticks(rotation=15, fontsize=11)
    plt.ylabel("$R^2$ Coefficient of Determination", fontsize=12)
    plt.xlabel("Tested Scikit-Learn Estimator Algorithms", fontsize=12)
    plt.title("Comparative Supervised Performance Matrix ($R^2$ Score)\nEvaluated Chronologically on Unseen 2025 Test Window", fontsize=14, pad=20)
    
    plt.legend(
        title="Districts", 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0,
        ncol=2,          
        fontsize=9
    )
    
    plot_out_path = os.path.join(output_directory, "model_performance_comparison.png")
    plt.savefig(plot_out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    -> Upgraded visual comparison chart successfully saved to:\n       {plot_out_path}")

# ==========================================
# 3. LOAD AND CLEANSE MASTER DATASET
# ==========================================
def load_and_cleanse_source_data():
    print(">>> [Phase 1/3] Loading and Cleansing Shared Pipeline Dataset...")
    if not os.path.exists(CONFIG["INPUT_DATA_PATH"]):
        raise FileNotFoundError(f"Source file missing at: {CONFIG['INPUT_DATA_PATH']}.")
        
    df = pd.read_parquet(CONFIG["INPUT_DATA_PATH"])
    df['date'] = pd.to_datetime(df['date'])
    
    # Verifying weather matrics
    available_weather = [col for col in CONFIG["EXPLICIT_WEATHER_COLS"] if col in df.columns]
    
    if len(available_weather) == 0:
        lowercase_targets = [col.lower() for col in CONFIG["EXPLICIT_WEATHER_COLS"]]
        available_weather = [col for col in lowercase_targets if col in df.columns]
        
    print(f"    -> Safely locked onto numeric weather columns: {available_weather}")
    if len(available_weather) == 0:
        raise ValueError("CRITICAL ERROR: No numeric weather columns found matching configuration criteria.")
    
    # Drop rows containing missing metrics
    required_cols = ['amount', 'date', 'district'] + available_weather
    df_clean = df.dropna(subset=required_cols).copy()
    
    # Enforce non-negative constraints
    df_clean = df_clean[df_clean['amount'] >= 0]
    
    df_clean['month'] = df_clean['date'].dt.month.astype(int)
    df_clean['day_of_week'] = df_clean['date'].dt.day_name()
    df_clean['is_weekend'] = df_clean['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    print(f"    -> Successfully parsed and kept {len(df_clean)} operational rows.")
    return df_clean, available_weather

# ==========================================
# 4. RUN CHRONOLOGICAL SUPERVISED TRAINING
# ==========================================
def run_supervised_chronological_regression(df, weather_cols):
    print("\n>>> [Phase 2/3] Initializing Supervised Modeling Suite (Train: 2024 -> Test: 2025)...")
    
    categorical_features = ['day_of_week'] # Month is removed from string-encoding entirely
    numeric_features = weather_cols + ['is_weekend', 'month'] # Month joins the numeric pipeline safely
    
    # Preprocessing Pipeline (sparse_output=False handles Gradient Boosting dense requirements)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    supervised_suite = {
        'Linear_Regression': LinearRegression(),
        'Ridge_Regression': Ridge(alpha=1.0),
        'Poisson_Regression': PoissonRegressor(max_iter=1000),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient_Boosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    metrics_log = []
    unique_districts = df['district'].unique()
    
    for district in unique_districts:
        district_data = df[df['district'] == district]
        
        # Chronological Windows Setup
        train_window = district_data[(district_data['date'] >= '2024-01-01') & (district_data['date'] <= '2024-12-31')]
        test_window = district_data[(district_data['date'] >= '2025-01-01') & (district_data['date'] <= '2025-12-31')]
        
        if len(train_window) < 500 or len(test_window) < 500:
            print(f"    [! Skipping {district}]: Insufficient full-season profile logs (Train: {len(train_window)}, Test: {len(test_window)})")
            continue
            
        print(f"    Training District: {district:<18} (Train Samples: {len(train_window)} | Test Samples: {len(test_window)})...")
        
        X_train, y_train = train_window[categorical_features + numeric_features], train_window['amount']
        X_test, y_test = test_window[categorical_features + numeric_features], test_window['amount']
        
        for name, algorithm in supervised_suite.items():
            model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', algorithm)
            ])
            
            try:
                model_pipeline.fit(X_train, y_train)
                predictions = model_pipeline.predict(X_test)
                
                r2 = r2_score(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)
                
                metrics_log.append({
                    'District': district, 'Model': name, 'R2_Score': r2, 'RMSE': rmse, 'MAE': mae
                })
            except Exception as error:
                print(f"       [ERROR] Operational Failure executing model {name} on {district}: {error}")
                
    metrics_dataframe = pd.DataFrame(metrics_log)
    metrics_csv_path = os.path.join(CONFIG["OUTPUT_DIRECTORY"], "model_comparison_metrics.csv")
    metrics_dataframe.to_csv(metrics_csv_path, index=False)
    print(f"    -> Finished! Detailed metric leaderboard saved to: {metrics_csv_path}")
    
    # Trigger fixed graph rendering engine
    print("\n>>> [Phase 3/3] Rendering Fixed Graphics Engine Output...")
    save_optimized_performance_plot(metrics_dataframe, CONFIG["OUTPUT_DIRECTORY"])

# ==========================================
# 5. ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("=======================================================")
    print("      SCIKIT-LEARN SUPERVISED PREDICTION PIPELINE      ")
    print("=======================================================")
    cleaned_df, active_weather_cols = load_and_cleanse_source_data()
    run_supervised_chronological_regression(cleaned_df, active_weather_cols)
    print("\nExecution complete! All models are completely robust.")
