import pandas as pd
import sys
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('df31.csv', header=0, index_col=0, squeeze=True)
X = series.values
size = int(len(X) * 1)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(1,2):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	history.append(yhat)
	print('predicted=%f' % (yhat))

df=pd.DataFrame(predictions)
df.to_csv("00.csv")