from keras.layers import Embedding, CuDNNLSTM, Bidirectional, Dense, CuDNNGRU
from keras.initializers import Constant
from keras import Sequential
import keras.backend as K
import numpy as np

EMBEDDING_DIM = 300
INPUT_LENGTH = 


class Model:
    def __init__(self, tokenizer, embedding_matrix,
                 rnn_units=50,
                 bidirectional=True,
                 rnn_type='lstm',
                 show_summary=True):
        rnn_types = {
            'lstm': CuDNNLSTM,
            'gru': CuDNNGRU
        }
        rnn_type = rnn_types[rnn_type]

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        num_words = len(tokenizer.word_index) + 1
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=INPUT_LENGTH,
                                    trainable=False)
        model = Sequential()
        model.add(embedding_layer)
        if bidirectional:
            model.add(Bidirectional(rnn_type(rnn_units)))
        else:
            model.add(rnn_type(rnn_units))
        model.add(Dense(num_words, activation='softmax'))
        if show_summary:
            model.summary()

        self.model = model
        self.tokenizer = tokenizer

    def train(self, X, y, epochs=5, batch_size=32, callbacks=[]):
        model = self.model
        # compile network
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[_perplexity])
        # fit network
        model.fit(X, y, 
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=1, 
                  shuffle=True,
                  validation_split=0.2,
                  callbacks=callbacks)

    def predict(self, first_word, n_words, B=3):
        tokenizer = self.tokenizer
        model = self.model
        in_text, result = first_word, first_word
        encoded = get_encoded(in_text, tokenizer)
        beam_sequences_scores = [[[encoded], 0]]

        while len(result) < n_words:
            all_candidates = []
            beam_sequences_scores = beam_step(beam_sequences_scores)
            assert len(beam_sequences_scores) == B
            for seq_score in beam_sequences_scores:
                seq_scores = beam_step(seq_score)
                all_candidates.append(seq_scores)
            beam_sequences_scores = sorted(all_candidates, reverse=True, key=lambda seq, score: score)[:B]
            assert len(beam_sequences_scores) == B
            result, _= beam_sequences_scores[0]
        
        words = [get_word(token) for token in result]
        return ' '.join(words)
        
        
def get_encoded(text, tokenizer):
    encoded = self.tokenizer.texts_to_sequences([text])[0]
    encoded = np.array(encoded)
    return encoded

def get_word(index):
    for word, idx in self.tokenizer.word_index.items():
         if idx == index:
            return word


def _perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity
