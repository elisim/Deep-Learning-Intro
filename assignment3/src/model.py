import warnings
warnings.filterwarnings("ignore")
import keras.layers as KL
from keras.initializers import Constant
from keras import Sequential
import keras.backend as K
import numpy as np
from keras.backend import epsilon
from keras_layer_normalization import LayerNormalization
from keras import regularizers
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 300
INPUT_LENGTH = 1
MELODY_VEC_LENGTH = 150
MELODY_CNN_VEC_LENGTH = 128


class LyricsOnlyModel:
    def __init__(self, tokenizer, embedding_matrix,
                 rnn_units=50,
                 bidirectional=True,
                 rnn_type='lstm',
                 dropout=0.3,
                 show_summary=True,
                 train_embedding=True):
        rnn_types = {
            'lstm': KL.CuDNNLSTM,
            'gru': KL.CuDNNGRU
        }
        rnn_type = rnn_types[rnn_type]

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        num_words = len(tokenizer.word_index) + 1
        embedding_layer = KL.Embedding(num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=INPUT_LENGTH,
                                    trainable=train_embedding)
        model = Sequential()
        model.add(embedding_layer)
        model.add(KL.Dropout(dropout))
        if bidirectional:
            model.add(KL.Bidirectional(rnn_type(rnn_units)))
        else:
            model.add(rnn_type(rnn_units))
        model.add(LayerNormalization())
        model.add(KL.Dense(num_words, kernel_regularizer=regularizers.l2(0.1), activation='softmax'))
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
                  validation_split=0.1,
                  callbacks=callbacks)

    def predict(self, first_word, n_words):
        in_text, result = first_word, first_word
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            encoded = np.array(encoded)
            
            words_probs = self.model.predict_proba(encoded, verbose=0)[0]
            
            # get 2 arrays of probs and word_tokens
            words_probs_enu = list(enumerate(words_probs))
            words_probs_sorted = sorted(words_probs_enu, key=lambda x: x[1], reverse=True) # sorting in descending order
            words_tokens, words_probs = list(zip(*words_probs_sorted))
            # normalizre to sum 1 
            words_probs = np.array(words_probs, dtype=np.float64)
            words_probs /= words_probs.sum().astype(np.float64)
            word_token = np.random.choice(words_tokens, p=words_probs)
            
            # map predicted word index to word
            out_word = get_word(word_token, self.tokenizer)
            # append to input
            in_text, result = out_word, result + ' ' + out_word
        return result      

    def predict_beam(self, first_word, n_words, B=3):
        tokenizer = self.tokenizer
        in_text, result = first_word, [first_word]
        encoded = get_encoded(in_text, tokenizer)
        beam_sequences_scores = [[encoded, 0]]

        while len(result) < n_words:
            all_candidates = []
            beam_sequences_scores = self.beam_step(beam_sequences_scores, B)
            for seq_score in beam_sequences_scores:
                seq_scores = self.beam_step([seq_score], B)
                all_candidates.append(seq_scores)
            flatten = lambda lst: [item for sublist in lst for item in sublist]
            beam_sequences_scores = sorted(flatten(all_candidates), reverse=True, key=lambda tup: tup[1])[:B]
            result, _ = beam_sequences_scores[0]
        
        words = [get_word(token, self.tokenizer) for token in result]
        return ' '.join(words)

    def beam_step(self, beam_sequences_scores, B):            
        all_candidates = []
        for seq, score in beam_sequences_scores: # for each sequence
            # predict top B words
            seq_pad = pad_sequences([[sample] for sample in seq], maxlen=INPUT_LENGTH)
            words_probs = self.model.predict_proba(seq_pad, verbose=0)[0]
            words_probs_enu = list(enumerate(words_probs))
            words_probs_sorted = sorted(words_probs_enu, key=lambda x: x[1], reverse=True) # sorting in descending order
            top_b_words_probs = words_probs_sorted[:B] # top B words with max probability
            # for each prob in top B words, create a candidate
            for token, prob in top_b_words_probs: 
                word_token = token
                candidate = [np.append(seq, word_token), score + np.log(prob + epsilon())] # todo: word_token 
                all_candidates.append(candidate)
        # take candidates with max score
        beam_sequences_scores = sorted(all_candidates, reverse=True, key=lambda tup: tup[1])[:B]
        return beam_sequences_scores


class LyricsMelodyModel:
    def __init__(self, tokenizer, embedding_matrix,
                 rnn_units=50,
                 bidirectional=True,
                 rnn_type='lstm',
                 dropout=0.3,
                 show_summary=True,
                 train_embedding=True):
        rnn_types = {
            'lstm': KL.CuDNNLSTM,
            'gru': KL.CuDNNGRU
        }
        rnn_type = rnn_types[rnn_type]

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        num_words = len(tokenizer.word_index) + 1
        embedding_layer = KL.Embedding(num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=INPUT_LENGTH,
                                    trainable=train_embedding)
        lyrics_input = KL.Input(shape=(INPUT_LENGTH,))
        melody_input = KL.Input(shape=(MELODY_VEC_LENGTH,))

        #         melody = Flatten()(melody_input)
        lyrics = embedding_layer(lyrics_input)
        lyrics = KL.Flatten()(lyrics)
        lyrics = KL.Dropout(dropout)(lyrics)
        combined = KL.Concatenate()([lyrics, melody_input])
        combined = KL.Reshape((1, EMBEDDING_DIM + MELODY_VEC_LENGTH))(combined)
        combined = rnn_type(rnn_units)(combined)
        if bidirectional:
            combined = KL.Bidirectional(combined)
        #         combined = LayerNormalization()(combined)
        combined = KL.Dense(num_words, kernel_regularizer=regularizers.l2(0.1), activation='softmax')(combined)
        model = Model(inputs=[lyrics_input, melody_input], outputs=[combined])

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
                  validation_split=0.1,
                  callbacks=callbacks)

    def predict(self, first_word, song, n_words):
        in_text, result = first_word, first_word
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            encoded = np.array(encoded)

            words_probs = self.model.predict([[encoded], [song]], verbose=0)[0]

            # get 2 arrays of probs and word_tokens
            words_probs_enu = list(enumerate(words_probs))
            words_probs_sorted = sorted(words_probs_enu, key=lambda x: x[1],
                                        reverse=True)  # sorting in descending order
            words_tokens, words_probs = list(zip(*words_probs_sorted))
            # normalizre to sum 1
            words_probs = np.array(words_probs, dtype=np.float64)
            words_probs /= words_probs.sum().astype(np.float64)
            word_token = np.random.choice(words_tokens, p=words_probs)

            # map predicted word index to word
            out_word = get_word(word_token, self.tokenizer)
            # append to input
            in_text, result = out_word, result + ' ' + out_word
        return result

    
class LyricsMelodyCNNModel:
    def __init__(self, tokenizer, embedding_matrix,
                 rnn_units=50,
                 bidirectional=True,
                 rnn_type='lstm',
                 dropout=0.3,
                 show_summary=True,
                 train_embedding=True,
                 n_filters=20):
        rnn_types = {
            'lstm': KL.CuDNNLSTM,
            'gru': KL.CuDNNGRU
        }
        rnn_type = rnn_types[rnn_type]

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        num_words = len(tokenizer.word_index) + 1
        embedding_layer = KL.Embedding(num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=INPUT_LENGTH,
                                    trainable=train_embedding)
        lyrics_input = KL.Input(shape=(INPUT_LENGTH,))
        melody_input = KL.Input(shape=(MELODY_CNN_VEC_LENGTH,MELODY_CNN_VEC_LENGTH, 1))

        # lyrics 
        lyrics = embedding_layer(lyrics_input)
        lyrics = KL.Flatten()(lyrics)
        lyrics = KL.Dropout(dropout)(lyrics)
        
        # melody CNN: [CONV + RELU => BATCH NORN] x 2 => MaxPool => Faltten  
        melody = KL.Conv2D(n_filters, (3, 3), padding="same", activation='relu')(melody_input)
        melody = KL.BatchNormalization()(melody)
        melody = KL.Conv2D(n_filters, (3, 3), padding="same", activation='relu')(melody)
        melody = KL.BatchNormalization()(melody)
        melody = KL.MaxPooling2D()(melody)
        melody = KL.Flatten()(melody)
        
        # concat
        combined = KL.Concatenate()([lyrics, melody])
        combined = KL.Reshape((1, 82220))(combined)
        combined = rnn_type(rnn_units)(combined)
        if bidirectional:
            combined = KL.Bidirectional(combined)
        #         combined = LayerNormalization()(combined)
        combined = KL.Dense(num_words, kernel_regularizer=regularizers.l2(0.1), activation='softmax')(combined)
        model = Model(inputs=[lyrics_input, melody_input], outputs=[combined])

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
                  validation_split=0.1,
                  callbacks=callbacks)

    def predict(self, first_word, song, n_words):
        in_text, result = first_word, first_word
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            encoded = np.array(encoded)

            words_probs = self.model.predict([[encoded], [song]], verbose=0)[0]

            # get 2 arrays of probs and word_tokens
            words_probs_enu = list(enumerate(words_probs))
            words_probs_sorted = sorted(words_probs_enu, key=lambda x: x[1],
                                        reverse=True)  # sorting in descending order
            words_tokens, words_probs = list(zip(*words_probs_sorted))
            # normalizre to sum 1
            words_probs = np.array(words_probs, dtype=np.float64)
            words_probs /= words_probs.sum().astype(np.float64)
            word_token = np.random.choice(words_tokens, p=words_probs)

            # map predicted word index to word
            out_word = get_word(word_token, self.tokenizer)
            # append to input
            in_text, result = out_word, result + ' ' + out_word
        return result
  

def get_encoded(text, tokenizer):
    encoded = tokenizer.texts_to_sequences([text])[0]
    encoded = np.array(encoded)
    return encoded


def get_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word


def _perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity
