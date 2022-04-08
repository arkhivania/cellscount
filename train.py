import argparse
import os
from datagen import DataGenerator
import cfpnetm
import uf_loss_functions
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--trainData', type=str, required=True)
parser.add_argument('--outputModelFolder', type=str, required=True)

args = parser.parse_args()

def get_ritems(set: str):
    folder = os.path.join(args.trainData, set)
    image_files = filter(lambda x: x.endswith("_mask.png"), os.listdir(folder))
    r_items = [os.path.join(set, x.replace("_mask.png", "")) for x in image_files]
    return r_items

# Datasets
partition = {"train" : get_ritems("train"), "validate" : get_ritems("validate")}
labels = {}

for i,(k,v) in enumerate(partition.items()):
    label = 0
    for x in v:
        labels[x] = label
        label += 1

dim = (128, 128)

training_generator = DataGenerator(partition['train'], 
    labels, args.trainData, 
    useAugmentation=True, useAugmentationFlip=True, augmentFactor=100, 
    dim=dim, n_classes=1)

validation_generator = DataGenerator(partition['validate'], 
    labels, args.trainData, 
    useAugmentation=False, useAugmentationFlip=True, augmentFactor=50, 
    dim=dim, n_classes=1)

slicesParameters = {
    'filtersFactor': 1,
    'firstElementFiltersFactor' : 1,
    'nclasses' : 1
}
input = tf.keras.Input((dim[0], dim[1], 1))
model = cfpnetm.CFPNetM(input, slicesParameters)
my_loss = uf_loss_functions.tversky_loss(delta=0.3)
optimizer = Adam()
mtrcs = metrics=['mse', 'accuracy']
model.compile(optimizer=optimizer, loss = my_loss, metrics=mtrcs)

print(model.summary())

earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint(args.outputModelFolder, verbose=1, save_best_only=True)

# Train model on dataset
model.fit(training_generator,
            validation_data=validation_generator,
            use_multiprocessing=False, 
            callbacks = [earlystopper, checkpointer],
            epochs=1000)

