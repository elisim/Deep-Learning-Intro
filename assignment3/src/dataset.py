import os
import itertools

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
    splitted_line = line.split(',')
    return {'artist': splitted_line[0], 'song_name': splitted_line[1], 'lyrics': ''.join(splitted_line[2:])}


def prepare_train_data():
    midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH,DATA_PATH, MIDI_PATH))]

    with open(os.path.join(ROOT_PATH, DATA_PATH, LYRICS_TRAIN)) as fh:
        train_lines = fh.read().splitlines()

    parsed_songs = []
    for song_line in train_lines:
        parsed_songs.append(parse_input_line(song_line))

    parsed_songs = list(itertools.chain.from_iterable(parsed_songs))

    for i,song in enumerate(parsed_songs):
        midi_file_path = '{}_-_{}.mid'.format(song['artist'].replace(' ', '_'), song['song_name'].replace(' ', '_'))
        if sum([1 for filename in midi_files_list if midi_file_path[:-4].replace('\\', '') in filename]) > 0:
            parsed_songs[i]['midi_path'] = os.path.join(ROOT_PATH,DATA_PATH, midi_file_path)
        else:
            print('song {} doesnt have a midi'.format(song))

    return parsed_songs

