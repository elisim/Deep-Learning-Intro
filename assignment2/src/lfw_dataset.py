"""
Labeled Faces in the Wild dataset of face photographs.
"""
import pathlib
from google_drive_downloader import GoogleDriveDownloader as gdd
from os.path import join, isdir, exists
from os import listdir, remove, path
from shutil import move
import requests
from imageio import imread
import numpy as np
import keras


root = '../data/lfw2'
train_info_url = 'http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt'
test_info_url = 'http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt'

np.random.seed(84)

IMAGES_DIM = 250
VGG_IMAGES_DIM = 224


def load_data(val_size=0.2):
    root='../data/lfw2' # TODO: delete after finish with Hyperas
    val_size=0.2 # TODO: delete after finish with Hyperas

    if not (isdir(root) and exists(root)):
        _download()
    else:
        print("Data already exist")

    same_train_paths, diff_train_paths = _extract_samples_paths(train_info_url)
    same_test_paths, diff_test_paths = _extract_samples_paths(test_info_url)

    # shuffle train samples and split train to train and val
    same_train_paths = np.random.permutation(same_train_paths)
    diff_train_paths = np.random.permutation(diff_train_paths)

    train_indexes = np.random.permutation(
        np.concatenate([np.zeros(int(len(same_train_paths)*val_size)), np.ones(int(len(same_train_paths)*(1-val_size)))])).astype(bool)
    val_indexes = ~train_indexes

    same_val_paths = same_train_paths[val_indexes]
    same_train_paths = same_train_paths[train_indexes]

    diff_val_paths = diff_train_paths[val_indexes]
    diff_train_paths = diff_train_paths[train_indexes]

    return same_train_paths, diff_train_paths, same_val_paths, diff_val_paths, same_test_paths, diff_test_paths


def n_images():
    data_root = pathlib.Path(root)
    all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
    return len(all_image_paths)


def n_entities():
    data_root = pathlib.Path(root)
    all_entities_paths = [str(path) for path in list(data_root.glob('*'))]
    return len(all_entities_paths)


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
    same_person_paths = [(path.join(root, per_1, '{}_{}.jpg'.format(per_1,index_1)), path.join(root, per_2, '{}_{}.jpg'.format(per_2, index_2))) for (per_1,index_1),(per_2, index_2) in same_person_paths]

    different_person_paths = [((sample[0], sample[1].zfill(4)), (sample[2], sample[3].zfill(4))) for sample in [line.split('\t') for line in file_text[amount_of_samples+1:-1]]]
    different_person_paths = [(path.join(root, per_1, '{}_{}.jpg'.format(per_1,index_1)), path.join(root, per_2,'{}_{}.jpg'.format(per_2, index_2))) for (per_1,index_1),(per_2, index_2) in different_person_paths]

    return same_person_paths, different_person_paths


def _load_image(path):
    return imread(path).reshape(IMAGES_DIM, IMAGES_DIM, 1) / 255  # normalize rgb to 0-1


def _load_image_vgg(path):
    """
    Load an image to be ready for the VGG16 model.
    """
    import cv2
    # load the image from the disk
    gray_image = imread(path)
    # transform the grayscale image to RGB
    backtorgb = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)
    # rezise the image to fit the VGG16 model shape
    resized_image = cv2.resize(backtorgb, dsize=(VGG_IMAGES_DIM, VGG_IMAGES_DIM), interpolation=cv2.INTER_CUBIC)
    reshaped_image = resized_image.reshape(1,224,224,3)
    #reshaped_image = reshaped_image.astype('float64')
    # taken from the keras_vggface repository
    #reshaped_image[:,:,:,0] -= 93.5940
    #reshaped_image[:,:,:,1] -= 104.7624
    #reshaped_image[:,:,:,2] -= 129.1863     
    return reshaped_image


def perpare_triplets():
    pass
    

class LFWDataLoader(keras.utils.Sequence):
    def __init__(self, same_paths, diff_paths, batch_size=32, dim=(IMAGES_DIM, IMAGES_DIM), channels=1, load_image_func=_load_image, shuffle=False, use_allocated_pairs=True, use_worst_pairs=True, size_allocated_pairs=12, model=None):
        
        self.dim = dim
        self.channels = channels
        self.load_image_func = load_image_func
        self.same_paths = same_paths
        self.diff_paths = diff_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_worst_pairs = use_worst_pairs
        self.size_worst_pairs = size_worst_pairs
        self.best_pairs = best_pairs
        self.model = model
        self.indexes = np.arange(len(self.same_paths))
        
        if use_allocated_pairs:
            if model is None:
                raise Exception("When using use_worst_pairs=True you need to pass a model to use to check the samples during training")
            if ((batch_size - size_allocated_pairs)% 2) != 0:
                raise(Exception('(batch_size - size_worst_pairs) need to be dividable by 2'))
            
            self.worst_pairs_X = np.empty((2, size_allocated_pairs, *self.dim, self.channels))
            self.worst_pairs_y = np.empty(size_allocated_pairs, dtype=float)
            self.batch_size = self.batch_size - self.size_allocated_pairs
            
            self.X_pairs = np.empty((2, len(self.diff_paths)*2, *self.dim, self.channels))
            self.y_pairs = np.empty(len(self.diff_paths)*2, dtype=float)
            
            for index in range(len(self.diff_paths)):
                self.X_pairs[0, index] = self.load_image_func(self.same_paths[index][0])
                self.X_pairs[1, index] = self.load_image_func(self.same_paths[index][1])    
                self.X_pairs[0, index + len(self.diff_paths)] = self.load_image_func(self.diff_paths[index][0])
                self.X_pairs[1, index + len(self.diff_paths)] = self.load_image_func(self.diff_paths[index][1])
                self.y_pairs[index] = 0
                self.y_pairs[index + len(self.diff_paths)] = 1
        else:
            if (batch_size % 2) != 0:
                raise(Exception('batch_size need to be dividable by 2'))
        if len(same_paths) != len(diff_paths):
            raise(Exception('we should have the same amount of paths for similar images and different images'))


        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.same_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
           
        if self.use_allocated_pairs:
            y_pred = self.model.predict([self.X_pairs[0], self.X_pairs[1]])
            if self.use_worst_pairs:
                ids_of_worst_pairs = np.argsort(np.abs(y_pred.T[0] - self.y_pairs))[-self.size_allocated_pairs:]
            else:
                ids_of_worst_pairs = np.argsort(np.abs(y_pred.T[0] - self.y_pairs))[:self.size_allocated_pairs]
                        
            for i in range(self.size_worst_pairs):
                self.worst_pairs_X[0, i] = self.X_pairs[0, ids_of_worst_pairs[i]]
                self.worst_pairs_X[1, i] = self.X_pairs[1, ids_of_worst_pairs[i]]
                self.worst_pairs_y[i] = self.y_pairs[ids_of_worst_pairs[i]]
            

    def generate_batch(self, image_indexes):
        if self.use_worst_pairs:
            X = np.empty((2, self.batch_size + self.size_allocated_pairs, *self.dim, self.channels), dtype=float)
            y = np.empty(self.batch_size + self.size_allocated_pairs, dtype=float)
        else:
            X = np.empty((2, self.batch_size, *self.dim, self.channels), dtype=float)
            y = np.empty(self.batch_size, dtype=float)

        index = 0
        for id in image_indexes:
            X[0, index] = self.load_image_func(self.same_paths[id][0])
            X[1, index] = self.load_image_func(self.same_paths[id][1])
            y[index] = 0
            index += 1

        for id in image_indexes:
            X[0, index] = self.load_image_func(self.diff_paths[id][0])
            X[1, index] = self.load_image_func(self.diff_paths[id][1])
            y[index] = 1
            index += 1

        if self.use_allocated_pairs:
            for i in range(self.size_allocated_pairs):
                X[0, index] = self.worst_pairs_X[0, i]
                X[1, index] = self.worst_pairs_X[1, i]
                y[index] = self.worst_pairs_y[i]
                index += 1

        perm_test = np.random.permutation(y.shape[0])
        X[0, ] = X[0, perm_test, ]
        X[1, ] = X[1, perm_test, ]
        y = y[perm_test]

        return X, y

    def __len__(self):
        return int(np.floor((len(self.same_paths) + len(self.diff_paths)) / self.batch_size))

    def __getitem__(self, batch_index):
        images_indexes = self.indexes[batch_index * int(self.batch_size/2):(batch_index + 1) * int(self.batch_size/2)]

        X, y = self.generate_batch(images_indexes)
        return [X[0,], X[1,]], y

