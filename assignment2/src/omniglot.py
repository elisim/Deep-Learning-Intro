import requests
import numpy as np
from os.path import isdir, exists
from os import makedirs, remove
from zipfile import ZipFile
from imageio import imread

DATASET_URL = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
EVALUATION_DATASET_URL = 'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'

root = '../data/omniglot'

np.random.seed(84)


def load_data():
    if not (isdir(root) and exists(root)):
        _download()
    else:
        print("Data already exist")

def _download():
    zip_dest = root + '.zip'
    print("Started Downloading omniglot...")

    r = requests.get(DATASET_URL)
    with open(zip_dest, 'wb') as fh:
        fh.write(r.content)

    makedirs(root)
    with ZipFile(zip_dest, 'r') as zipObj:
        zipObj.extractall(path=root)

    remove(zip_dest)
    print("Data downloaded successfully")


def _load_images(paths):
    images = []
    for path_1, path_2 in paths:
        images_dim = imread(path_1).shape[0]
        images.append((imread(path_1).reshape(images_dim, images_dim, 1) / 255,
                       imread(path_2).reshape(images_dim, images_dim, 1) / 255))
    return images
