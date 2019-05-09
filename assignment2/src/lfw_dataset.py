"""
Labeled Faces in the Wild dataset of face photographs.
"""
import pathlib
import random
from google_drive_downloader import GoogleDriveDownloader as gdd
from os.path import join, isdir, exists
from os import listdir, remove
from shutil import move

root='../data/lfw2'

def load_data():
    if not (isdir(root) and exists(root)):
        _download()
    else:
        print("Data already exist")
    # TODO:
    # add return of x_train, y_train, x_test, y_test

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
    
