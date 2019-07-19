from tqdm import tqdm
import string
import gensim
import nltk
import pickle
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle
import itertools
import random
from keras.utils import to_categorical

ROOT_PATH = "."
DATA_PATH = "Data"
MIDI_PATH = "midi_files"
LYRICS_TRAIN = "lyrics_train_set.csv"
LYRICS_TEST = "lyrics_test_set.csv"


WORDS_VECTORS_DIR = 'word_vectors/'
LYRICS_DIR = 'Data/'
GLOVE_DIR = os.path.join(WORDS_VECTORS_DIR, 'glove.6B')
TEXT_DATA = os.path.join(LYRICS_DIR, 'unified_lyrics_dump.txt')

MAX_SEQUENCE_LENGTH = 1 # During each step of the training phase, your architecture will receive as input one word of the lyrics.
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

def parse_input_line(line):
    # For the case we have more than one song in a line
    if '&  &  &' in line:
        subsongs = line.split('&  &  &')
        parsed_subsongs = []
        for song_line in subsongs:
            parsed_subsongs.append(parse_lyrices_line(song_line))
        return parsed_subsongs
    else:
        return [parse_lyrices_line(line)]


def parse_lyrices_line(line):
    splitted_line = line.split(',')
    return {'artist': splitted_line[0], 'song_name': splitted_line[1], 'lyrics': ''.join(splitted_line[2:]),
            'X': [], 'y': []}


def prepare_train_data():
    # extract list of midi files
    midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH,DATA_PATH, MIDI_PATH))]

    # read the lyrics from the train file
    with open(os.path.join(ROOT_PATH, DATA_PATH, LYRICS_TRAIN)) as fh:
        train_lines = fh.read().splitlines()

    # parse the songs
    parsed_songs = []
    for song_line in train_lines:
        parsed_songs.append(parse_input_line(song_line))

    # in case of 2 songs in one line, flatten the songs
    parsed_songs = list(itertools.chain.from_iterable(parsed_songs))

    # get midi path for each song
    for i,song in enumerate(parsed_songs):
        midi_file_path = '{}_-_{}.mid'.format(song['artist'].replace(' ', '_'), song['song_name'].replace(' ', '_'))
        if sum([1 for filename in midi_files_list if midi_file_path[:-4].replace('\\', '') in filename]) > 0:
            parsed_songs[i]['midi_path'] = os.path.join(ROOT_PATH,DATA_PATH, midi_file_path)

    # add special tokens
    for i, song in enumerate(parsed_songs):
        # change & sign in <EOL> and remove redundent dash
        parsed_songs[i]['lyrics'] = ' '.join(nltk.word_tokenize(parsed_songs[i]['lyrics']))

        # add <EOS> in the end of each song and change & to </s>
        parsed_songs[i]['lyrics'] = parsed_songs[i]['lyrics'].replace('&', '.')
        parsed_songs[i]['lyrics'] += " eos"

    # split lyrics by windows size
    for i, song in enumerate(tqdm(parsed_songs, total=len(parsed_songs))):
        splitted_lyrics = [token for token in nltk.word_tokenize(parsed_songs[i]['lyrics']) if token == '.' or token not in string.punctuation]
        for j in range(len(splitted_lyrics) - 1):
            parsed_songs[i]['X'].append(splitted_lyrics[j])
            parsed_songs[i]['y'].append(splitted_lyrics[j+1])
        parsed_songs[i]['X'] = np.array(parsed_songs[i]['X'])
        parsed_songs[i]['y'] = np.array(parsed_songs[i]['y'])

    return parsed_songs


def load_vocab():
    X, _ = load_data()
    return list(set(X.flatten()))


def load_data():
    parsed_songs = prepare_train_data()

    X = np.hstack([song['X'] for song in parsed_songs])
    y = np.hstack([song['y'] for song in parsed_songs])
    return X, y


def load_tokenized_data():
    X,y = load_data()

    all_songs_words = ' '.join(X.flatten()) + ' eos'
    tokenizer = init_tokenizer(all_songs_words)

    X = [lst[0] for lst in tokenizer.texts_to_sequences(X)]
    y = [lst[0] for lst in tokenizer.texts_to_sequences(y)]
    y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)

    return X, y, tokenizer


def extract_embedding_weights():
    X, y, tokenizer = load_tokenized_data()

    # prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1

    pretrained_embeddings = load_pretrained_embedding()
    embedding_matrix, not_found = prepare_embedding_matrix(num_words, EMBEDDING_DIM, word_index, pretrained_embeddings)
    return embedding_matrix

def prepare_embedding_matrix(num_of_words, embedding_dim, word_index, pretrained_embeddings):
    embedding_matrix = np.zeros((num_of_words, embedding_dim))
    not_found = []
    for word, i in word_index.items():  #TODO: check also word in capitlal (for word2vec)
        word_encode = word.encode()
        embedding_vector = pretrained_embeddings.get(word_encode)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found.append(word)  #TODO: solve unknown word in pretrained_embeddings (words with ')

    return embedding_matrix, not_found


def load_pretrained_embedding(embedding_type='glove'):
    local_pickle_file = os.path.join(WORDS_VECTORS_DIR, f'{embedding_type}_embeddings.pickle')
    if not os.path.exists(local_pickle_file):
        embeddings_index = {}
        if embedding_type == 'glove':
            with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), 'rb') as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, 'f', sep=' ')
                    embeddings_index[word] = coefs
        else:
            #TODO: implement word2vec vectors extraction
            pass
        with open(os.path.join(WORDS_VECTORS_DIR, f'{embedding_type}_embeddings.pickle'), 'wb') as f:
            pickle.dump(embeddings_index, f)
        return embeddings_index

    with open(os.path.join(WORDS_VECTORS_DIR, f'{embedding_type}_embeddings.pickle'), 'rb') as f:
        pretrained_embeddings = pickle.load(f)

    return pretrained_embeddings


def init_tokenizer(text):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts([text])
    return tokenizer

