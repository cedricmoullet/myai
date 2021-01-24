import requests
import io
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

# Load data

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

