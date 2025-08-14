#IMPORTING THE PANDAS EXTENTION

import pandas #https://colab.research.google.com/drive/1hftHUQd1bL7sdW-iNGalrGyBwFbq5yN7datareader.data as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import yfinance as yf
import pytz
from datetime import datetime as dt

#NOW DOWNLOAD THE STOCK DATA
tz = pytz.timezone("America/New_York")
start_date = tz.localize(dt(2022,1,1))
end_date = tz.localize(dt(2022,6,14))


df = yf.download(['AAPL'], start=start_date, end=end_date, auto_adjust=True)

#df.to_csv("stocks.csv")

#NOW CREATING THE DATASET
df["Diff"] = df.Close.diff()
df["SMA_2"] = df.Close.rolling(2).mean()

df["Force_Index"] = df["Close"] * df["Volume"]

df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)

df = df.drop(
   ["Open", "High", "Low", "Close", "Volume", "Diff"],
   axis=1,
).dropna()

#df.head()
# # # # print(df)
X = df.drop(["y"], axis=1).values
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.2,
   shuffle=False,
)
df.head()
X_train.shape
input_shape=(X_train.shape[1], 1)
print(input_shape)
y
#Training
clf = make_pipeline(StandardScaler(), MLPClassifer(random_state=0, shuffle=False))
clf.fit(
    X_train,
    y_train,
)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred

y_test
