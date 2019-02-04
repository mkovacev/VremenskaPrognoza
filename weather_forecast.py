import numpy as np
import csv
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras


def split(sequences, n):
	X, Y = list(), list()
	print(sequences.shape[0])
	for i in range(sequences.shape[0]):
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

def train(Xtrain, Ytrain, numOfDays, numOfEpochs):
	n_features = Xtrain.shape[2]
	# define model
	model = Sequential()
	model.add(LSTM(80, activation='relu',return_sequences=True, input_shape=(numOfDays, n_features)))
	model.add(LSTM(40, activation='relu', dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(n_features))
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	# fit model
	history = model.fit(Xtrain, Ytrain, validation_split=0.1, epochs=numOfEpochs, verbose=1, batch_size = 100)

	model.save("model.h5")
	return history

def test(Xtest, Ytest, history, x): 

	model = keras.models.load_model("model.h5")
	# Evaluation
	score = model.evaluate(Xtest, Ytest)
	print('Test score:', score)
	
	# Plot training and validation accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	axes = plt.gca()
	axes.set_ylim([0,1])
	plt.yticks(np.arange(0, 1, 0.1))
	axes.set_xlim([0,3])
	plt.xticks(np.arange(0, x, 1))
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train_acc', 'val_acc'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	axes = plt.gca()
	axes.set_ylim([0,0.3])
	plt.yticks(np.arange(0, 0.3, 0.1))
	axes.set_xlim([0,3])
	plt.xticks(np.arange(0, x, 1))
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train_loss', 'val_loss'], loc='upper left')
	plt.show()

def loadData():
	data = np.array(list(csv.reader(open("weather.csv", "r"), delimiter=","))).astype("float")

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

	return database


def main():

	database = loadData()

	numOfDays = 10

	X, Y = split(database, numOfDays)

	Xtrain = X[:5000, :]
	Ytrain = Y[:5000, :]
	Xtest = X[5000:, :]
	Ytest = Y[5000:, :]

	history = train(Xtrain, Ytrain, numOfDays, 20)
	test(Xtest, Ytest, history, 20)

if __name__ == "__main__":
    main()
