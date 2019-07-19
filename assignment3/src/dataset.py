import os
import itertools
import numpy as np
from tqdm import tqdm
import re
import string
import gensim
import nltk

from sklearn.preprocessing import OneHotEncoder

ROOT_PATH = "."
DATA_PATH = "Data"
MIDI_PATH = "midi_files"
LYRICS_TRAIN = "lyrics_train_set.csv"
LYRICS_TEST = "lyrics_test_set.csv"

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

    # # prepare one hot encoding of the lyrics
    # all_words = np.array(list(set(itertools.chain.from_iterable([song['lyrics'].split() for song in parsed_songs])))).reshape(-1, 1)
    # encoder = OneHotEncoder(sparse=False)
    # encoder.fit(all_words)
    #return parsed_songs, encoder

    return parsed_songs

# # Relevant if we are using OHE manually
# def get_encoded_word(enc, word):
#     return enc.transform(np.array([word]).reshape(-1,1)).flatten()


def load_vocab():
    X, _ = load_data()
    return list(set(X.flatten()))


def load_data():
    parsed_songs = prepare_train_data()

    X = np.hstack([song['X'] for song in parsed_songs])
    y = np.hstack([song['y'] for song in parsed_songs])
    return X, y


def dump_lyrics_to_file():
    with open(os.path.join(ROOT_PATH, DATA_PATH, 'unified_lyrics_dump.txt'), 'w') as fh:
        X, _ = load_data()
        fh.write(' '.join(X.flatten()) + ' eos')


def get_embedding_weights(embedding_type='glove'):
    if embedding_type == 'glove':
        print('Not implemented')
    else:
        word_model = gensim.models.KeyedVectors.load_word2vec_format('pre_trained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        word_model.wv

