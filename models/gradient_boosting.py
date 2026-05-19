from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from weather_data2 import merged
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# defining dates as separate variables
merged["month"] = merged["date"].dt.month
merged["day_of_week"] = merged["date"].dt.dayofweek
merged["is_weekend"] = (
    merged["day_of_week"] >= 5
).astype(int)

merged = pd.get_dummies(
    merged,
    columns=["district"],
    drop_first=True
)

merged.info()

exclude = [
    "amount",
    "date",
    "day",
    "latitude",
    "longitude",
    "site_id",
    "PIXEL_ID",
    "type",
    "source_month",
    "direction",
    "direction_name",
    "site_nr",
    "name_site",
    "domain",
    "path_nr",
    "council",
    "installation_date",
    "pixel_id",
    "short_wave_from_sky",
    "evapotrans_ref",
    "interval"
]

features = [
    col for col in merged.columns
    if col not in exclude
]

y = merged["amount"]
x = merged[features]

x = x.fillna(x.median())

#test-train split based on the year (2024 = train, 2025 = test)
train = merged[merged["date"] < "2025-01-01"]
test = merged[merged["date"] >= "2025-01-01"]

X_train = train[features]
y_train = train["amount"]

X_test = test[features]
y_test = test["amount"]

###XGBoost model
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

# evaluate XGBoost
mse = mean_squared_error(y_test, xgb_preds)
print("XGBoost MSE:", mse)

rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print("XGBoost RMSE:", rmse)

mae = mean_absolute_error(y_test, xgb_preds)
print("XGBoost MAE:", mae)

r2 = r2_score(y_test, xgb_preds)
print("XGBoost R²:", r2)

# LightGBM model
lgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)


# evaluate LightGBM
lgbm_preds = np.asarray(lgbm_preds)

mse = mean_squared_error(y_test, lgbm_preds)
print("LightGBM MSE:", mse)

rmse = np.sqrt(mean_squared_error(y_test, lgbm_preds))
print("LightGBM RMSE:", rmse)

mae = mean_absolute_error(y_test, lgbm_preds)
print("LightGBM MAE:", mae)

r2 = r2_score(y_test, lgbm_preds)
print("LightGBM R²:", r2)

#compare the two
print("XGBoost R²:", r2_score(y_test, xgb_preds))
print("LightGBM R²:", r2_score(y_test, lgbm_preds))