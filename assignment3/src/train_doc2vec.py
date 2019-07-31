import joblib
import numpy as np
from midi_processing import prepare_doc2vec

def main():
    # load the entire melody, harmony and drums samples we extracted from the songs
    data = joblib.load('midi_preprocess_data.jblib')

    # Extract each one of the samples type seperatly 
    X_drums = np.vstack([i for i in data[0] if len(i.shape) > 1])
    X_melody = np.vstack([i for i in data[1] if len(i.shape) > 1])
    X_harmony = np.vstack([i for i in data[2] if len(i.shape) > 1])

    # Train the Doc2Vec models
    drums_model = prepare_doc2vec(X_drums)
    melody_model = prepare_doc2vec(X_melody)
    harmony_model = prepare_doc2vec(X_harmony)

    # Dump the models for future loading
    joblib.dump('midi_preprocess/models/drums_model')
    joblib.dump('midi_preprocess/models/melody_model')
    joblib.dump('midi_preprocess/models/harmony_model')

if __name__ == __main__:
    main()