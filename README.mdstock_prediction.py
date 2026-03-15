# stock_price_prediction_history.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

# --------------------------
# Suppress FutureWarnings
# --------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------
# 1. Download historical stock data
# --------------------------
symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2025-11-07"

# Use .history() to get a clean DataFrame for a single stock
df = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=True)

# Check data
print("Data sample:")
print(df.head())

# --------------------------
# 2. Feature engineering
# Predict 'Close' price based on previous day's Close and Volume
# --------------------------
df['Prev_Close'] = df['Close'].shift(1)
df['Prev_Volume'] = df['Volume'].shift(1)
df = df.dropna()

X = df[['Prev_Close', 'Prev_Volume']]
y = df['Close']

# --------------------------
# 3. Train/test split (time-series)
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --------------------------
# 4. Scale features
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 5. Train model
# --------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --------------------------
# 6. Predict & evaluate
# --------------------------
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# --------------------------
# 7. Plot actual vs predicted
# --------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Close')
plt.plot(y_pred, label='Predicted Close')
plt.title(f"{symbol} — Actual vs Predicted Close Price")
plt.xlabel('Test data index')
plt.ylabel('Price')
plt.legend()
plt.show()

# --------------------------
# 8. Predict next-day close price
# --------------------------
latest_row = df.iloc[-1]
feat = np.array([[latest_row['Close'], latest_row['Volume']]])  # shape (1,2)
feat_scaled = scaler.transform(feat)
pred_price = model.predict(feat_scaled)[0]
print(f"Predicted next-day close price for {symbol}: {pred_price:.2f}")

