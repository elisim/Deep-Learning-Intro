import numpy as np
import pretty_midi
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from src.consts import *


def check_if_melody(instrument, silence_threshold=0.7, mono_threshold=0.8, fs=10):
    """
    Check if the given instrument is Melody, Harmony or too silence

    :param instrument: the object that contain the note information
    :param silence_threshold: the threshold that above it the instrument considired to be too quiet.
    :param mono_threshold: the threshold that above it the instrument considered to be a Melody
    :param fs: the rate to sample from the midi
    :return: True - the instrument is considered as melody, False - the instrument considered as harmony, -1 - the
    instrument considered as too quiet.
    """
    # Extract all of the notes of the instrument
    piano_roll = instrument.get_piano_roll(fs=fs)

    # extract the timeframes the contain notes
    timeframes_with_notes_indexes = np.unique(np.where(piano_roll != 0)[1])
    piano_roll_notes = piano_roll[:, timeframes_with_notes_indexes]

    n_timeframes = piano_roll.shape[1]
    n_notes = piano_roll_notes.shape[1]
    n_silence = n_timeframes - n_notes
    n_mono = np.count_nonzero(np.count_nonzero(piano_roll_notes > 0, axis=0) == 1)

    # check if instrument is too quiet
    if silence_threshold <= float(n_silence)/n_timeframes:
        return -1

    if mono_threshold <= float(n_mono)/n_notes:
        return True

    return False


def number_to_note(number):
    """
    Extract note name from note number

    :param number: index of note
    :return: note name or "r" for rest
    """
    if number == 128:
        return 'r'
    else:
        return pretty_midi.note_number_to_name(number)


def extract_notes_from_melody(instrument, window_size=50, fs=5, training_output=True):
    """
    Extract the notes strings from the melody instrument

    :param instrument: the object that contain the note information
    :param window_size: size of output "sentence"
    :param fs: the rate to sample from the midi
    :param training_output: if True - extract sentences in window_size size, if False - extract all of the notes in one
                            list
    :return: notes in string format
    """

    # Extract all of the notes of the instrument
    instrument_timeframes = instrument.get_piano_roll(fs=fs)

    # find where is the first note
    melody_start = np.min(np.where((np.sum(instrument_timeframes, axis=0) > 0)))
    melody_piano_roll = instrument_timeframes[:, melody_start:]

    # TODO: filter all timeframes after the last note

    # ignore the velocity of the melody
    melody_piano_roll = (melody_piano_roll > 0).astype(float)

    # add an index for the rest notes, and assign 1 in those indexes
    rests = np.sum(melody_piano_roll, axis=0)
    rests = (rests == 0).astype(float)
    melody_piano_roll = np.insert(melody_piano_roll, 128, rests, axis=0)

    # if training_output=True, split the samples to windows otherwise extract one list with all of the notes
    if training_output:
        X = []
        for i in range(0, melody_piano_roll.shape[1] - window_size):
            window = melody_piano_roll[:, i:i + window_size]
            X.append([number_to_note(np.where(note==1)[0][0]) for note in window.T])
        return np.array(X)
    else:
        return [number_to_note(np.where(note == 1)[0][0]) for note in melody_piano_roll.T]


def extract_notes_from_harmony(instrument, window_size=200, fs=5, training_output=True):
    """
    Extract the notes strings from the melody instrument

    :param instrument: the object that contain the note information
    :param window_size: size of output "sentence"
    :param fs: the rate to sample from the midi
    :param training_output: if True - extract sentences in window_size size, if False - extract all of the notes in one
                            list
    :return: notes in string format
    """

    # Extract all of the notes of the instrument
    instrument_timeframes = instrument.get_piano_roll(fs=fs)

    # find where is the first note
    harmony_start = np.min(np.where((np.sum(instrument_timeframes, axis=0) > 0)))
    harmony_piano_roll = instrument_timeframes[:, harmony_start:]

    # TODO: filter all timeframes after the last note

    # ignore the velocity of the melody
    harmony_piano_roll = (harmony_piano_roll > 0).astype(float)

    # add an index for the rest notes, and assign 1 in those indexes
    rests = np.sum(harmony_piano_roll, axis=0)
    rests = (rests == 0).astype(float)
    harmony_piano_roll = np.insert(harmony_piano_roll, 128, rests, axis=0)

    # if training_output=True, split the samples to windows otherwise extract one list with all of the notes
    if training_output:
        X = []
        for i in range(0, harmony_piano_roll.shape[1] - window_size):
            window = harmony_piano_roll[:, i:i + window_size]
            X.append(['-'.join([number_to_note(note_num) for note_num in np.where(note==1)[0]]) for note in window.T])
        return np.array(X)
    else:
        return ['-'.join([number_to_note(note_num) for note_num in np.where(note==1)[0]]) for note in harmony_piano_roll.T]


def prepare_doc2vec(X):
    """
    Trainig a Doc2Vec model where doc == song

    :param X: The samples
    :return: Doc2Vec model
    """
    #TODO: chane parameters of doc2vec
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
    model = Doc2Vec(documents, vector_size=50, window=5, min_count=1, workers=4)
    return model


def prepare_midi_embeddings_dataset(fs=10):
    # prepare 3 different samples - for drums, for harmony and for the melody

    X_drums = []
    X_melody = []
    X_harmony = []

    list_midi_files = os.listdir(os.path.join(DATA_PATH, MIDI_PATH))
    for midi_file in tqdm(list_midi_files, total=len(list_midi_files)):
        # load the midi file
        try:
            midi_obj = pretty_midi.PrettyMIDI(os.path.join(DATA_PATH, MIDI_PATH, midi_file))
        except Exception as e:
            print('Error in loading {} file: {}'.format(midi_file, e))
            continue

        # parse each one of the instruments
        for inst in midi_obj.instruments:

            # if that drums we need to have special handling
            if inst.is_drum:
                inst.is_drum = False
                # check that notes give information
                if np.count_nonzero(inst.get_piano_roll(fs=fs)) == 0:
                    continue

                X = extract_notes_from_harmony(inst, fs=fs)
                X_drums.append(X)

                inst.is_drum = True
                continue

            # if its not drums and there is no notes - dont use it
            if np.count_nonzero(inst.get_piano_roll(fs=fs) != 0) == 0:
                continue

            # now check if that instrument is melody or harmony
            is_melody = check_if_melody(inst)
            if is_melody == True:
                X = extract_notes_from_melody(inst, fs=fs)
                X_melody.append(X)
            elif is_melody == False:
                X = extract_notes_from_harmony(inst, fs=fs)
                X_harmony.append(X)
            else:
                # Instrument is too quiet
                continue

    return X_drums, X_melody, X_harmony


def get_song_vector(midi_path, models, fs=10):
    # load the doc2vec models
    drum_model = models['drums']
    melody_model = models['melody']
    harmony_model = models['harmony']

    # extract the notes from the instruments in the midi_file
    midi_obj = pretty_midi.PrettyMIDI(midi_path)
    melody_notes = []
    harmony_notes = []
    drums_notes = []

    for inst in midi_obj.instruments:

        # if that drums we need to have special handling
        if inst.is_drum:
            inst.is_drum = False
            # check that notes give information
            if np.count_nonzero(inst.get_piano_roll(fs=fs)) == 0:
                continue

            drums_notes += extract_notes_from_harmony(inst, fs=fs, training_output=False)

            inst.is_drum = True
            continue

        # if its not drums and there is no notes - dont use it
        if np.count_nonzero(inst.get_piano_roll(fs=fs) != 0) == 0:
            continue

        # now check if that instrument is melody or harmony
        is_melody = check_if_melody(inst)
        if is_melody == True:
            melody_notes += extract_notes_from_melody(inst, fs=fs, training_output=False)
        elif is_melody == False:
            harmony_notes += extract_notes_from_harmony(inst, fs=fs, training_output=False)
        else:
            # Instrument is too quiet
            continue

    drums_embedding = drum_model.infer_vector(drums_notes)
    melody_embedding = melody_model.infer_vector(melody_notes)
    harmony_embedding = harmony_model.infer_vector(harmony_notes)

    return np.hstack([drums_embedding, melody_embedding, harmony_embedding])



# def extract_midi_piano_roll(midi_path, resize_to_size=None, fs=10):
#     midi_obj = pretty_midi.PrettyMIDI(midi_path)
#     results = midi_obj.get_piano_roll(fs=fs)
#
#     if not resize_to_size is None:
#
#     return results