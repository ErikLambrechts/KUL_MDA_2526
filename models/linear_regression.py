import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

merged = pd.read_parquet('data/MDA_Final_Dataset.parquet')

daily = merged.groupby(
    ["district", "date"]
).agg({
    "amount": "sum",
    "temperature": "mean",
    "precipitation": "mean",
    "humidity": "mean",
    "wind_speed": "mean",
    "sunshine_duration": "mean"
}).reset_index()

merged["month"] = pd.to_datetime(
    merged["date"]).dt.month

merged["day_of_week"] = pd.to_datetime(
    merged["date"]).dt.dayofweek

merged["weekend"] = (
    merged["day_of_week"] >= 5).astype(int)

merged = pd.get_dummies(
    merged,
    columns=["district"],
    drop_first=True
)

daily["log_amount"] = np.log1p(
    daily["amount"]
)

results = {}

districts = daily["district"].unique()

for district in districts:

    district_data = daily[
        daily["district"] == district
    ].copy()

features = [
    "temperature",
    "precipitation",
    "humidity",
    "wind_speed",
    "sunshine_duration",
    "month",
    "weekend"
]

model_data = district_data[
    features + ["log_amount"]
].dropna()

x = model_data[features]
y = model_data["log_amount"]

x = x.apply(pd.to_numeric)
x = x.astype(float)
y = y.astype(float)


model = LinearRegression().fit(x, y)
results[district] = {
    "model": model,
    "features": features,
    "r_squared": model.score(x, y),
    "coefficients": model.coef_,
    "intercept": model.intercept_
}

#assumption checking
##residuals vs fitted values plot
fitted = model.predict(x)
residuals = y - fitted

plt.scatter(fitted, residuals) 
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show() #residuals aren't scattered randomly around zero, indicating potential heteroscedasticity

#assumptions for the log transformed model
##residuals vs fitted values plot
fitted = model.predict(x)
residuals = y - fitted

plt.scatter(fitted, residuals) 
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show() #doesn't look much different than the previous one, still some heteroscedasticity because of the funnel shape 

##normality of residuals


## multicollinearity
vif = pd.DataFrame()

vif["feature"] = x.columns

vif["VIF"] = [
    variance_inflation_factor(x.values, i)
    for i in range(x.shape[1])
]

print(vif) #temperature vatiables are highly correlated with y

##influential observations
#influence = model.get_influence()

cooks = influence.cooks_distance[0]

plt.scatter(
    range(len(cooks)),
    cooks
)

plt.show() #quite small? <0.001