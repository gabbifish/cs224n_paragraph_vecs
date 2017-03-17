# use fit after running LSTM on all inputs.

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

print('Loading original doc2vec model...')
doc2vec_model = model = Doc2Vec.load('../doc2vec/small_wiki_subset.en.doc2vec.model')

print('Build sequential model fed by LSTM...')
in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
model.add(Dense(hidden_neurons, in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

print('model compiled!')

print('Training Sequential/LSTM model...')
model.fit(doc2vec_model.docvecs, None, batch_size=batch_size, epochs=15,
          validation_data=(x_test, y_test)) # do not want validation_data! will run clustering tests!

print('Saving model')
model.save('lstm_seq.small_wiki_subset.model')
