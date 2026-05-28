import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pure Scikit-Learn Unsupervised Data Mining Toolkit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. GLOBAL SYSTEM CONFIGURATION & DIRECTORIES
# ==========================================
CONFIG = {
    "INPUT_DATA_PATH": "./KUL_MDA_2526/data/MDA_Final_Dataset.parquet",
    "OUTPUT_DIRECTORY": r"./KUL_MDA_2526/data",
    "EXPLICIT_WEATHER_COLS": ['temp_max', 'temp_min', 'temp_avg', 'precip_quantity', 'humidity_relative', 'pressure', 'sun_duration', 'short_wave_from_sky', 'evapotrans_ref']
}

os.makedirs(CONFIG["OUTPUT_DIRECTORY"], exist_ok=True)

# ==========================================
# 2. LOAD AND CLEANSE MASTER DATASET
# ==========================================
print(">>> [Phase 1/3] Loading Shared Master Dataset...")
if not os.path.exists(CONFIG["INPUT_DATA_PATH"]):
    raise FileNotFoundError(f"Source file missing at: {CONFIG['INPUT_DATA_PATH']}.")
    
df = pd.read_parquet(CONFIG["INPUT_DATA_PATH"])
df['date'] = pd.to_datetime(df['date'])

# Dynamically discover weather columns
ignored_cols = {'amount', 'date', 'district', 'site_id', 'direction', 'geometry', 'type'}
active_weather_cols = [col for col in df.columns if col not in ignored_cols]
print(f"    -> Dynamically discovered weather indicators: {active_weather_cols}")

# Drop rows missing crucial modeling variables
df_clean = df.dropna(subset=['amount', 'date', 'district'] + active_weather_cols).copy()
df_clean = df_clean[df_clean['amount'] >= 0]

# Feature Engineering
df_clean['is_weekend'] = df_clean['date'].dt.dayofweek.isin([5, 6]).astype(int)

# ==========================================
# 3. GENERATE DESCRIPTIVE TABLES
# ==========================================
print("\n>>> [Phase 2/3] Engineering Descriptive Statistical Summary Tables...")
reporting_columns = ['amount'] + active_weather_cols + ['is_weekend']
descriptive_table = df_clean[reporting_columns].describe()

csv_out_path = os.path.join(CONFIG["OUTPUT_DIRECTORY"], "descriptive_statistics.csv")
descriptive_table.to_csv(csv_out_path)
print(f"    -> Success! Descriptive profiling stored to: {csv_out_path}")

# ==========================================
# 4. UNSUPERVISED LEARNING (Clustering, PCA, Anomalies)
# ==========================================
print("\n>>> [Phase 3/3] Activating Unsupervised Architecture...")

# Isolate columns for multi-variate structural tracking
unsupervised_features = ['amount'] + list(active_weather_cols)
X_matrix = df_clean[unsupervised_features].copy()

# Scale features perfectly for distance calculations
scaler = StandardScaler()
X_scaled_matrix = scaler.fit_transform(X_matrix)

# A. K-Means Clustering
print("    -> Executing K-Means Cluster Structural Sorting (3 Groups)...")
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['cluster_label'] = kmeans_model.fit_predict(X_scaled_matrix)

# B. PCA Dimensionality Reduction
print("    -> Compressing dimensions via PCA for visualization...")
pca_transformer = PCA(n_components=2)
pca_coordinates = pca_transformer.fit_transform(X_scaled_matrix)
df_clean['PCA1'] = pca_coordinates[:, 0]
df_clean['PCA2'] = pca_coordinates[:, 1]
explained_var = sum(pca_transformer.explained_variance_ratio_) * 100

# PLOT GENERATION: Render the clusters visually
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='PCA1', y='PCA2', hue='cluster_label', palette='Set1', alpha=0.4, edgecolor=None)
plt.title(f"Unsupervised Cycling Profiles: Structural K-Means Groupings\nTotal Variance Captured: {explained_var:.1f}%")
plt.xlabel("Principal Component 1 (PCA1)")
plt.ylabel("Principal Component 2 (PCA2)")
plt.tight_layout()

pca_plot_path = os.path.join(CONFIG["OUTPUT_DIRECTORY"], "pca_clusters_visualization.png")
plt.savefig(pca_plot_path, dpi=150)
plt.close()
print(f"    -> PCA Cluster visualization saved to: {pca_plot_path}")

# C. Anomaly Detection (Isolation Forest)
print("    -> Initiating Outlier Sweep using Isolation Forest (Top 2%)...")
outlier_detector = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
df_clean['anomaly_flag'] = outlier_detector.fit_predict(X_scaled_matrix)

# D. EXPORT LABELED DATAFRAME
final_output_path = os.path.join(CONFIG["OUTPUT_DIRECTORY"], "MDA_Analyzed_Dataset.parquet")
df_clean.to_parquet(final_output_path, index=False)
print(f"    -> Finished! Labeled master dataset written to:\n       {final_output_path}")


print("\n=======================================================")
print("          UNSUPERVISED RUN COMPLETE RESULTS            ")
print("=======================================================")
print("\n--- Structural Group Cluster Traffic Averages (amount) ---")
print(df_clean.groupby('cluster_label')['amount'].mean().to_string())
print("\n--- Structural Data Anomaly Distribution Counts ---")
print(df_clean['anomaly_flag'].value_counts().to_string().replace(" 1", "   Normal Traffic Profile ( 1)").replace("-1", "   Flagged Statistical Outlier (-1)"))
print("=======================================================")
