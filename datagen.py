import numpy as np
import os
import tensorflow as tf
import numpy
import imageio
from tensorflow.keras.utils import to_categorical
import scipy.ndimage as ndi
from math import cos, sin, pi
import random as pyr
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

def transform_image(image, angle=0.0, cval = 0, scale=1.0, aniso=1.0, translation=(0, 0), order=1, flip = 0.25):
    dx, dy = translation
    scale = 1.0/scale
    c = cos(angle)
    s = sin(angle)
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)
    w, h, channels = image.shape
    c = np.array([w, h]) / 2.0
    d = c - np.dot(m, c) + np.array([dx * w, dy * h])

    r = np.zeros((w, h, channels), dtype=image.dtype)
    for j in range(channels):
        tr = ndi.affine_transform(image[:,:,j], m, offset=d, order=order, mode='nearest', output=image.dtype, cval=cval)
        
        if flip > 0.5:
            tr = np.fliplr(tr)
        r[:,:,j] = tr

    return r

def random_transform(translation=(-0.15, 0.15), rotation=(-20, 20), scale=(-0.1, 0.1), aniso=(-0.1, 0.1)):
    flip = pyr.uniform(0.0, 1.0)
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    angle = pyr.uniform(*rotation)
    angle = angle * pi / 180.0
    scale = 10**pyr.uniform(*scale)
    aniso = 10**pyr.uniform(*aniso)
    return dict(angle=angle, scale=scale, aniso=aniso, translation=(dx, dy), flip=flip)

def random_scale_shift_rotate(image, label, augmentRotation, useFlip = False):
    t = random_transform(rotation=(-augmentRotation, augmentRotation))
    if useFlip == False:
        t['flip'] = 0.0

    transformed_image = transform_image(image, cval=-1024.0, order=1, **t)
    transformed_label = transform_image(label, cval=0, order=1, **t) # для label тоже применяем линейную интерполяцию, так как это уже categorical
    return (transformed_image, transformed_label)


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, 
                 inputFolder,
                 dim=(32,32), batch_size=32, n_channels=1,
                 n_classes=10, shuffle=True, useAugmentation = False, useAugmentationFlip = False,
                 augmentFactor = 3, augmentRotation = 60.0):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.inputFolder = inputFolder
        self.useAugmentation = useAugmentation
        self.useAugmentationFlip = useAugmentationFlip
        self.augmentFactor = augmentFactor
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.augmentRotation = augmentRotation

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.list_IDs) * (self.augmentFactor if self.useAugmentation else 1)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k % len(self.list_IDs)] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs) * (self.augmentFactor if self.useAugmentation else 1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_classes))
        
        def worker(i, ID):
            # Store sample
            basePath = os.path.join(self.inputFolder, ID)
            sliceImage = imageio.imread(basePath + ".png")
            X[i,:,:,0] = sliceImage            

            sliceMask = imageio.imread(basePath + "_mask.png")[:,:,1]
            sliceMask = (sliceMask >= 255) * sliceMask
            Y[i,:,:,0] = sliceMask/255.0

            if self.useAugmentation:
                X[i,:,:,:], Y[i,:,:,:] = random_scale_shift_rotate(X[i,:,:,:], Y[i,:,:,:], augmentRotation=self.augmentRotation, useFlip=self.useAugmentationFlip)

        futures = []
        for i, ID in enumerate(list_IDs_temp):
            futures.append(self.executor.submit(worker, i = i, ID = ID))
        
        done, futures = wait(futures, timeout=None, return_when=ALL_COMPLETED)
        if len(futures) > 0:
            raise Exception('futures trouble')
        return X, Y

