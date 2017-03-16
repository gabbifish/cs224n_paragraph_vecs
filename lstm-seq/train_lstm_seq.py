# use fit after running LSTM on all inputs.
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
model.add(Dense(hidden_neurons, in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
