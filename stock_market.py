

# Import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load TCS stock data (2018–2024)
df = yf.download('TCS.NS', start='2018-01-01', end='2024-12-31')
df = df.reset_index()

# Prepare data for Prophet
df = df[['Date', 'Close']]
df.columns = ['ds', 'y']

# Visualize stock trend
plt.figure(figsize=(12,6))
plt.plot(df['ds'], df['y'])
plt.title('TCS Stock Price (2018–2024)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

# Create and train model
model = Prophet()
model.fit(df)

# Create future dates (next 365 days)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Forecast of TCS Stock Price")
plt.show()

# Plot components (trend, seasonality)
model.plot_components(forecast)
plt.show()

# Show last 5 predictions
print("Last 5 Predicted Values:")
print(forecast[['ds', 'yhat']].tail())
