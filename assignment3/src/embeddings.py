import warnings
warnings.filterwarnings("ignore")
from .dataset import load_tokenized_data
import gensim
import pickle
import os
import numpy as np

WORDS_VECTORS_DIR = 'word_vectors/'
EMBEDDING_DIM = 300
GLOVE_DIR = os.path.join(WORDS_VECTORS_DIR, 'glove.6B')


def extract_embedding_weights(embedding_type='glove'):
    X, y, tokenizer = load_tokenized_data()

    # prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1

    pretrained_embeddings = load_pretrained_embedding(embedding_type)
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
            word_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(WORDS_VECTORS_DIR, 'GoogleNews-vectors-negative300.bin'), binary=True)
            for word in word_model.vocab.keys():
                embeddings_index[word] = word_model[word]
        with open(local_pickle_file, 'wb') as f:
            pickle.dump(embeddings_index, f)
        return embeddings_index

    with open(local_pickle_file, 'rb') as f:
        pretrained_embeddings = pickle.load(f)

    return pretrained_embeddings
