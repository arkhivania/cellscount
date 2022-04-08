import argparse
import imageio
import numpy as np
from libtiff import TIFFfile

parser = argparse.ArgumentParser(description='Apply model to file')
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--channel', type=int, required=True)
args = parser.parse_args()

source_image = TIFFfile(args.input)
samples, sample_names = source_image.get_samples()
source = samples[0][args.channel]
source_size = source.shape

import tensorflow as tf

def my_loss(y_true,y_pred):
    return None

model = tf.keras.models.load_model(args.model, custom_objects = {
    'loss_function' : my_loss
})
size = (model.input.shape[1], model.input.shape[2])

res = np.zeros(source_size, dtype=np.float32)
modelInput = np.zeros((1, size[0], size[1], 1))

for y in range(0, source_size[0], size[0]):
    print("processing tile line: {}/{}".format(y, source_size[0]))
    for x in range(0, source_size[1], size[1]):
        max_y = source_size[0] if y + size[0] > source_size[0] else y + size[0]
        max_x = source_size[1] if x + size[1] > source_size[1] else x + size[1]
        ty = max_y-y
        tx = max_x-x
        modelInput[0,0:ty,0:tx,0] = source[y:max_y,x:max_x]
        segmBatchResult = model.predict(modelInput, batch_size=1)
        res[y:max_y, x:max_x] = np.maximum(res[y:max_y, x:max_x], segmBatchResult[0, 0:ty, 0:tx, 0])

imageio.imwrite("check_source.png", source)        
imageio.imwrite("check.png", res * 255)
