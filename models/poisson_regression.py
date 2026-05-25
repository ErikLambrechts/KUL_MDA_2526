import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from models.weather_data2 import merged

merged.head()

merged = pd.get_dummies(
    merged,
    columns=["district"],
    drop_first=True
)

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

print(
    x.select_dtypes(include=["bool"]).columns
)

bool_cols = x.select_dtypes(
    include=["bool"]
).columns

x[bool_cols] = x[bool_cols].astype(int)

model = make_pipeline(
    StandardScaler(),
    PoissonRegressor(max_iter=1000)
)

results = model.fit(x, y)

print(results.score(x, y))
print(model.named_steps['poissonregressor'].coef_)
print(model.named_steps['poissonregressor'].intercept_)
np.exp(model.named_steps['poissonregressor'].coef_)

#test-train split based on the year (2024 = train, 2025 = test)
train = merged[merged["date"] < "2025-01-01"]
test = merged[merged["date"] >= "2025-01-01"]

X_train = train[features]
y_train = train["amount"]

X_test = test[features]
y_test = test["amount"]

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Train:", model.score(X_train, y_train))
print("Test:", model.score(X_test, y_test))