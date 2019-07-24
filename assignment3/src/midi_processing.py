import numpy as np
import gensim
import pretty_midi
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from tqdm import tqdm
from src.dataset import prepare_train_data

MIDI_PATH = "Data/midi_files"

def check_if_melody(instrument, silence_threshold=0.7, mono_threshold=0.80):
    piano_roll = instrument.get_piano_roll(fs=10)
    timeframes_with_notes_indexes = np.unique(np.where(piano_roll != 0))
    piano_roll_notes = piano_roll[:, timeframes_with_notes_indexes]
    n_timeframes = piano_roll.shape[1]
    n_notes = piano_roll_notes.shape[1]
    n_silence = n_timeframes - n_notes
    n_mono = np.count_nonzero(np.count_nonzero(piano_roll_notes > 0, axis=0) == 1)

    if silence_threshold <= float(n_silence)/n_timeframes:
        return -1

    if mono_threshold <= float(n_mono)/n_notes:
        return True

    return False


def number_to_note(number):
    if number == 128:
        return 's'
    else:
        return pretty_midi.note_number_to_name(number)


def extract_sample_from_melody_all(instrument, fs=5):
    instrument_timeframes = instrument.get_piano_roll(fs=fs)
    melody_start = np.min(np.where((np.sum(instrument_timeframes, axis=0) > 0)))
    melody_piano_roll = instrument_timeframes[:, melody_start:]
    melody_piano_roll = (melody_piano_roll > 0).astype(float)
    rests = np.sum(melody_piano_roll, axis=0)
    rests = (rests != 1).astype(float)
    melody_piano_roll = np.insert(melody_piano_roll, 128, rests, axis=0)

    return [number_to_note(np.where(note==1)[0][0]) for note in melody_piano_roll.T]

def extract_sample_from_melody_windows(instrument, window_size=20, fs=5):
    instrument_timeframes = instrument.get_piano_roll(fs=fs)
    melody_start = np.min(np.where((np.sum(instrument_timeframes, axis=0) > 0)))
    melody_piano_roll = instrument_timeframes[:, melody_start:]
    melody_piano_roll = (melody_piano_roll > 0).astype(float)
    rests = np.sum(melody_piano_roll, axis=0)
    rests = (rests != 1).astype(float)
    melody_piano_roll = np.insert(melody_piano_roll, 128, rests, axis=0)

    X = []

    for i in range(0, melody_piano_roll.shape[1] - window_size):
        window = melody_piano_roll[:, i:i + window_size]
        X.append([number_to_note(np.where(note==1)[0][0]) for note in window.T])
    return np.array(X)

def prepare_doc2vec(X):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
    model = Doc2Vec(documents, vector_size=50, window=5, min_count=1, workers=4)
    return model


def prepare_midi_embeddings(fs=5):
    X_total = []
    for midi_file in tqdm(os.listdir(MIDI_PATH)[ : 20], total=20):
        try:
            midi_obj = pretty_midi.PrettyMIDI(os.path.join(MIDI_PATH,midi_file))
        except Exception as e:
            continue
        for inst in midi_obj.instruments:
            if inst.is_drum:
                continue
            if np.count_nonzero(inst.get_piano_roll(fs=fs) != 0) == 0:
                continue
            if check_if_melody(inst):
                X = extract_sample_from_melody_windows(inst, fs=fs)
                X_total.append(X)
    X_total = np.vstack(X_total)
    model = prepare_doc2vec(X_total)
    return model


