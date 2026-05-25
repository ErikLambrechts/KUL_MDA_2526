import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from models.weather_data2 import merged
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

print(type(x))
print(
    x.select_dtypes(include=["object"]).columns
)

print(
    x.select_dtypes(include=["bool"]).columns
)

bool_cols = x.select_dtypes(
    include=["bool"]
).columns

x[bool_cols] = x[bool_cols].astype(int)

x = sm.add_constant(x) #intercept
model = sm.OLS(y, x).fit()
print(model.summary())

#assumption checking
##residuals vs fitted values plot
fitted = model.fittedvalues
residuals = model.resid

plt.scatter(fitted, residuals) 
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show() #residuals aren't scattered randomly around zero, indicating potential heteroscedasticity

#normality of residuals
sns.histplot(residuals, kde=True)
plt.show() #seem to be mainly concentrated around 0 with some skewness 

sm.qqplot(residuals, line="45")
plt.show() #crazy pattern, not normal, need to use log transformation



#log-linear
merged["log_amount"] = np.log1p(
    merged["amount"]
)

y = merged["log_amount"]

model = sm.OLS(y, x).fit()
print(model.summary())

#assumptions for the log transformed model
##residuals vs fitted values plot
fitted = model.fittedvalues
residuals = model.resid

plt.scatter(fitted, residuals) 
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show() #doesn't look much different than the previous one, still some heteroscedasticity because of the funnel shape 

##normality of residuals
sns.histplot(residuals, kde=True)
plt.show() #looks the same as before

sm.qqplot(model.resid, line="45")
plt.show() #looks much better, doesn't follow the line perfectky but is close

## multicollinearity
vif = pd.DataFrame()

vif["feature"] = x.columns

vif["VIF"] = [
    variance_inflation_factor(x.values, i)
    for i in range(x.shape[1])
]

print(vif) #temperature vatiables are highly correlated with y

##influential observations
influence = model.get_influence()

cooks = influence.cooks_distance[0]

plt.scatter(
    range(len(cooks)),
    cooks
)

plt.show() #quite small? <0.001