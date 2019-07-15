import os
import itertools
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ROOT_PATH = ".."
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
    splitted_line = line.split(',');
    return {'artist': splitted_line[0], 'song_name': splitted_line[1], 'lyrics': ''.join(splitted_line[2:]), 'X': [],
            'y': []}


def prepare_train_data(window_size=10):
    midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH,DATA_PATH, MIDI_PATH))]

    with open(os.path.join(ROOT_PATH, DATA_PATH, LYRICS_TRAIN)) as fh:
        train_lines = fh.read().splitlines()

    parsed_songs = []
    for song_line in train_lines:
        parsed_songs.append(parse_input_line(song_line))

    parsed_songs = list(itertools.chain.from_iterable(parsed_songs))

    # get midi path for each song
    for i,song in enumerate(parsed_songs):
        midi_file_path = '{}_-_{}.mid'.format(song['artist'].replace(' ', '_'), song['song_name'].replace(' ', '_'))
        if sum([1 for filename in midi_files_list if midi_file_path[:-4].replace('\\', '') in filename]) > 0:
            parsed_songs[i]['midi_path'] = os.path.join(ROOT_PATH,DATA_PATH, midi_file_path)

    # add special tokens
    for i, song in enumerate(parsed_songs):
        # change & sign in <EOL>
        parsed_songs[i]['lyrics'] = parsed_songs[i]['lyrics'].replace('&', '<EOL>')

        # add <EOS> in the end
        parsed_songs[i]['lyrics'] += " <EOS>"

    # split lyrics by windows size
    for i, song in enumerate(parsed_songs):
        splitted_lyrics = parsed_songs[i]['lyrics'].split()
        for window in range(len(splitted_lyrics) - window_size):
            parsed_songs[i]['X'].append(splitted_lyrics[window: window + window_size])
            parsed_songs[i]['y'].append(splitted_lyrics[window + 1: window + window_size + 1])
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


def load_data(window_size=10):
    parsed_songs = prepare_train_data(window_size)

    X = np.concatenate([song['X'] for song in parsed_songs])
    y = np.concatenate([song['y'] for song in parsed_songs])
    return X, y

def apply_embedding():
    #TODO: implement embedding
    pass
