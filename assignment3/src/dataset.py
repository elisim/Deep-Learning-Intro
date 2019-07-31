import warnings
warnings.filterwarnings("ignore")
import string
import nltk
import os
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tqdm import tqdm
from src.midi_processing import get_song_vector, extract_midi_piano_roll
import joblib
from src.consts import *


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
    midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH, DATA_PATH, MIDI_PATH))]

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
            parsed_songs[i]['midi_path'] = os.path.join(ROOT_PATH, DATA_PATH, MIDI_PATH, midi_file_path)
    
    # remove songs without midi
    parsed_songs = [song for song in parsed_songs if 'midi_path' in song.keys()]

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
    X, _ = load_data(with_melody=False)
    return list(set(X.flatten())) + ['eos']


def load_data(with_melody=True, melody_type='doc2vec'):
    parsed_songs = prepare_train_data()

    X_midi = [(song['midi_path'], len(song['X'])) for song in parsed_songs]
    X = np.hstack([song['X'] for song in parsed_songs])
    y = np.hstack([song['y'] for song in parsed_songs])

    if with_melody:
        models = {name: joblib.load(os.path.join(DOC2VEC_MODELS_PATHS, f'{name}_model.jblib')) for name in
                  ['drums', 'melody', 'harmony']}

        songs = []
        X = [] 
        y = []
        for i, (midi_path, count) in tqdm(enumerate(X_midi), total=len(X_midi), desc='Loading the songs embedding'):
            try:
                if melody_type == 'doc2vec':
                    songs.append(np.array([get_song_vector(midi_path, models)]*count))
                else:
                    songs.append([extract_midi_piano_roll(midi_path)] * count)
                
                # We don't wont to load songs with problematic midi
                X.append(parsed_songs[i]['X'])
                y.append(parsed_songs[i]['y'])
            except Exception as e:
                print(r"Couldn't load {}, issue with the midi file.".format(midi_path))
                continue
        songs = np.vstack(songs)
        X = np.hstack(X)
        y = np.hstack(y)
        return X, y, songs
        
    return X, y


def load_tokenized_data(with_melody=False, melody_type='doc2vec', max_samples=-1):
    if with_melody:
        X, y, songs = load_data(with_melody=with_melody, melody_type=melody_type)
    else:
        X, y = load_data(with_melody=with_melody, melody_type=melody_type)

    all_songs_words = ' '.join(load_vocab())
    tokenizer = init_tokenizer(all_songs_words)

    X = [lst[0] for lst in tokenizer.texts_to_sequences(X)]
    y = [lst[0] for lst in tokenizer.texts_to_sequences(y)]
    y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)

    if max_samples != -1:
        X = X[:max_samples]
        y = y[:max_samples]
    
    if with_melody:
        if max_samples != -1:
            songs = songs[:max_samples, :]
        return X, y, tokenizer, songs
    else:
        return X, y, tokenizer


def init_tokenizer(text):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts([text])
    return tokenizer


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
