import pandas as pd
from weather_data2 import merged
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error)
import numpy as np
from sklearn.metrics import mean_squared_error

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

x.info()

x = x.fillna(x.median())

#test-train split based on the year (2024 = train, 2025 = test)
train = merged[merged["date"] < "2025-01-01"]
test = merged[merged["date"] >= "2025-01-01"]

X_train = train[features]
y_train = train["amount"]

X_test = test[features]
y_test = test["amount"]

#model 
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# cheching the performance of the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2) #r2 is negative, so the model is no good?

importance = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(importance) #might be irrelevant since the model didn't perform well