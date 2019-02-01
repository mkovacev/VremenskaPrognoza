import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def split(sequences, n):
	X, Y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end = i + n
		# check if we are beyond the dataset
		if end > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seqX, seqY = sequences[i:end, :], sequences[end, :]
		X.append(seqX)
		Y.append(seqY)
	return np.array(X), np.array(Y)


def main():

  data = np.array(list(csv.reader(open("weather.csv", "rb"), delimiter=","))).astype("float")

  months = data.T[0]
  monthsMin = months.min()
  monthsMax = months.max()
  months = np.interp(months, (months.min(), months.max()), (0, 1))

  maxTemp = data.T[1]
  maxTempMin = maxTemp.min()
  maxTempMax = maxTemp.max()
  maxTemp = np.interp(maxTemp, (maxTemp.min(), maxTemp.max()), (0, 1))

  minTemp = data.T[2]
  minTempMin = minTemp.min()
  minTempMax = minTemp.max()
  minTemp = np.interp(minTemp, (minTemp.min(), minTemp.max()), (0, 1))

  humidity = data.T[3]
  humidityMin = humidity.min()
  humidityMax = humidity.max()
  humidity = np.interp(humidity, (humidity.min(), humidity.max()), (0, 1))

  pressure = data.T[4]
  pressureMin = pressure.min()
  pressureMax = pressure.max()
  pressure = np.interp(pressure, (pressure.min(), pressure.max()), (0, 1))

  wind = data.T[5]
  windMin = wind.min()
  windMax = wind.max()
  wind = np.interp(wind, (wind.min(), wind.max()), (0, 1))

  months = months.reshape((len(months), 1))
  maxTemp = maxTemp.reshape((len(maxTemp), 1))
  minTemp = minTemp.reshape((len(minTemp), 1))
  humidity = humidity.reshape((len(humidity), 1))
  pressure = pressure.reshape((len(pressure), 1))
  wind = wind.reshape((len(wind), 1))

  database = np.hstack((months, maxTemp, minTemp, humidity, pressure, wind))

  numOfDays = 14

  X, Y = split(database, numOfDays)
  
  n_features = X.shape[2]

  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(numOfDays, n_features)))
  model.add(LSTM(100, activation='relu'))
  model.add(Dense(n_features))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(X, Y, epochs=50, verbose=1)
  # demonstrate prediction
  x_input = np.array([X[15]])
  print(x_input)
  x_input = x_input.reshape((1, numOfDays, n_features))
  y = model.predict(x_input, verbose=1)
  y[0][0] = np.interp(y[0][0], (0, 1), (monthsMin, monthsMax))
  y[0][1] = np.interp(y[0][1], (0, 1), (maxTempMin, maxTempMax))
  y[0][2] = np.interp(y[0][2], (0, 1), (minTempMin, minTempMax))
  y[0][3] = np.interp(y[0][3], (0, 1), (humidityMin, humidityMax))
  y[0][4] = np.interp(y[0][4], (0, 1), (pressureMin, pressureMax))
  y[0][5] = np.interp(y[0][5], (0, 1), (windMin, windMax))
  print(y[0])

  

if __name__ == "__main__":
    main()