# Original description: https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f
# Corresponding colab: https://colab.research.google.com/drive/1mUMS6A95yM-YEiQB8FN-CdBdlH5Z_ADG#scrollTo=VBXq-Wb_K1XR
import requests
import io
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import LSTM

# Load data
# Colab: https://colab.research.google.com/drive/1mUMS6A95yM-YEiQB8FN-CdBdlH5Z_ADG

path = 'https://drive.google.com/uc?export=download&id=1mQr7hY6yO88nv5SmLbYBRk14cJJ_FQnP'

s = requests.get(path).content
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(io.StringIO(s.decode('utf-8')),
                 header='infer',
                 delimiter=',',
                 parse_dates=['Date'],
                 date_parser=dateparse)

# Plot data
print(df.info())

ax = df.plot(x="Date", y="BTC")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_xlabel("Date")

# Create a pairplot
sns.pairplot(df,
             height=2.5,
             x_vars=['BTC', 'ETH', 'LTC'],
             y_vars=['BTC', 'ETH', 'LTC']
             )

#####################
# Normalization
#####################

scaler = MinMaxScaler()

btc_price = df.BTC.values.reshape(-1, 1)

scaled_btc = scaler.fit_transform(btc_price)

scaled_btc = scaled_btc[~np.isnan(scaled_btc)]

scaled_btc = scaled_btc.reshape(-1, 1) # really needed ??

#####################
# Preprocessing
#####################

SEQ_LEN = 240 # 10 days

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = preprocess(scaled_btc, SEQ_LEN, train_split = 0.95)

print(X_train.shape)
print(y_train.shape)

#####################
# Model
#####################

DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

model = keras.Sequential()

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))

model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))

model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))

model.add(Activation('linear'))

#####################
# Taining
#####################

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

BATCH_SIZE = 64

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#####################
# Prediction
#####################

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)

y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')

plt.show()

print("End")

