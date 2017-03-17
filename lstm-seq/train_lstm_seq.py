# use fit after running LSTM on all inputs.
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

print('Loading data...')

print('Build model...')
in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
model.add(Dense(hidden_neurons, in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=15,
          validation_data=(x_test, y_test)) # do not want validation_data! will run clustering tests!
