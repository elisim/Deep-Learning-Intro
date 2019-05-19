"""
Labeled Faces in the Wild dataset of face photographs.
"""
import pathlib
import random
from google_drive_downloader import GoogleDriveDownloader as gdd
from os.path import join, isdir, exists
from os import listdir, remove, path
from shutil import move
import requests
from imageio import imread
import numpy as np

root='../data/lfw2'
train_info_url = 'http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt'
test_info_url = 'http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt'

def load_data():
    if not (isdir(root) and exists(root)):
        _download()
    else:
        print("Data already exist")

    same_train_paths, diff_train_paths = _extract_samples_paths(train_info_url)
    same_test_paths, diff_test_paths = _extract_samples_paths(test_info_url)

    X_train_same = _load_images(same_train_paths)
    X_train_diff = _load_images(diff_train_paths)
    y_train = np.concatenate([np.ones(len(X_train_same)), np.zeros(len(X_train_diff))])
    X_train = np.concatenate([X_train_same, X_train_diff])
    X_train = [X_train[:,0], X_train[:, 1]]


    X_test_same = _load_images(same_test_paths)
    X_test_diff = _load_images(diff_test_paths)
    y_test = np.concatenate([np.ones(len(X_test_same)), np.zeros(len(X_test_diff))])
    X_test = np.concatenate([X_test_same, X_test_diff])
    X_test = [X_test[:, 0], X_test[:, 1]]

    return X_train, y_train, X_test, y_test

def size():
    data_root = pathlib.Path(root)
    all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
    random.shuffle(all_image_paths)
    return len(all_image_paths)
    
    
def _download():
    dest = root + '.zip'
    print("Started Downloading Labeled Faces in the Wild...")
    gdd.download_file_from_google_drive(file_id='1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp',
                                            dest_path=dest,
                                            unzip=True)
    for filename in listdir(join(root, 'lfw2')):
        move(join(root, 'lfw2', filename), join(root, filename))
    remove(dest)
    print("Data downloaded successfully")
    

def _extract_samples_paths(url):
    response = requests.get(url)
    file_text = response.text.split('\n')
    amount_of_samples = int(file_text[0])

    assert(len(file_text) == amount_of_samples*2 + 2)

    same_person_paths = [((sample[0], sample[1].zfill(4)), (sample[0], sample[2].zfill(4))) for sample in [line.split('\t') for line in file_text[1:amount_of_samples+1]]]
    same_person_paths = [(path.join('../data/lfw2/{}/{}_{}.jpg'.format(per_1,per_1,index_1)), path.join('../data/lfw2/{}/{}_{}.jpg'.format(per_2,per_2, index_2))) for (per_1,index_1),(per_2, index_2) in same_person_paths]

    different_person_paths = [((sample[0], sample[1].zfill(4)), (sample[2], sample[3].zfill(4))) for sample in [line.split('\t') for line in file_text[amount_of_samples+1:-1]]]
    different_person_paths = [(path.join('../data/lfw2/{}/{}_{}.jpg'.format(per_1,per_1,index_1)), path.join('../data/lfw2/{}/{}_{}.jpg'.format(per_2,per_2, index_2))) for (per_1,index_1),(per_2, index_2) in different_person_paths]

    return same_person_paths, different_person_paths


def _load_images(paths):
    images = []
    for path_1, path_2 in paths:
        images_dim = imread(path_1).shape[0]
        images.append((imread(path_1).reshape(images_dim, images_dim, 1)/255, imread(path_2).reshape(images_dim, images_dim, 1)/255))
    return images

