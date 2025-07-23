import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Date range for fetching stock data
start = '2010-01-01'
end = '2024-12-31'

# Ticker dictionary with names
ticker_dict = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BRK-B": "Berkshire Hathaway Inc. (Class B)",
    "V": "Visa Inc.",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "SBIN.NS": "State Bank of India",
    "RELIANCE.NS": "Reliance Industries Ltd"
}

# Streamlit UI
st.title('📈 Stock Trend Prediction Tool')

# Dropdown to select company
selected_name = st.selectbox("Select a Stock", options=list(ticker_dict.values()))
user_input = [k for k, v in ticker_dict.items() if v == selected_name][0]

# Fetch data from Yahoo Finance
df = yf.download(user_input, start=start, end=end)

# Validate data
if df.empty:
    st.error("No data available for the selected ticker.")
    st.stop()

# Display stats
st.subheader('Data from 2010 - 2024')
st.write(df.describe())

# Plot Closing Price
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
st.pyplot(fig)

# Plot with 100MA
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
st.pyplot(fig)

# Plot with 100MA and 200MA
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
st.pyplot(fig)

# Split data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

# Scale training data
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load pretrained model
model = load_model('keras_model.h5')

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_scaler = MinMaxScaler(feature_range=(0,1))
input_data = input_scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Reverse scaling
scale_factor = 1 / input_scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot predictions
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{user_input} Stock Price Prediction')
plt.grid(True)
plt.legend()
st.pyplot(fig2)
