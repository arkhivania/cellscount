import os

import numpy as np
import argparse
from scipy import misc
import imageio
import random
from skimage.transform import rescale, resize, downscale_local_mean
from libtiff import TIFFfile

random.seed(0)

parser = argparse.ArgumentParser(description='Prepare patch images')
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--outputFolder', type=str, required=True)
parser.add_argument('--tileSize', type=int, required=False, default=128)
parser.add_argument('--netSize', type=int, required=False, default=128)
parser.add_argument('--channel', type=int, required=True)

args = parser.parse_args()

source_image = TIFFfile(args.input)
samples, sample_names = source_image.get_samples()
# for image in source_image.iter_images():
source_image = samples[0][args.channel]

# source_image = imageio.imread(args.input)
for r in ["test", "train", "validate"]:
    folder = os.path.join(args.outputFolder, r)
    if not os.path.exists(folder):
        os.makedirs(folder)

tileSize = args.tileSize
for y in range(0, source_image.shape[0], tileSize):
    print("{} from {} complete".format(y//tileSize, source_image.shape[0]//tileSize))
    for x in range(0, source_image.shape[1], tileSize):
        if (x + tileSize < source_image.shape[1]) and \
            (y + tileSize < source_image.shape[0]):
            tileImg = source_image[y: y + tileSize, x:x + tileSize]

            targetFolder = "train"
            t = random.random()
            if t > 0.5:
                if t < 0.85:
                    targetFolder = "validate"
                else:
                    targetFolder = "test"
            targetFolder = os.path.join(args.outputFolder, targetFolder)
            pref = "{}_{}_".format(os.path.basename(args.input), args.channel)
            fname = "{}_{}_{}_img.png".format(pref, y, x)
            maskName = "{}_{}_{}_img_mask.png".format(pref, y, x)

            if args.netSize != tileSize:
                tileImg = resize(tileImg, 
                    (args.netSize, args.netSize), 
                    anti_aliasing=True, preserve_range=True, order=3).astype(np.uint8)
            
            mv = np.amax(tileImg)
            if mv > 0:
                imageio.imwrite(os.path.join(targetFolder, fname), tileImg)
                if not os.path.exists(os.path.join(targetFolder, maskName)):
                    imageio.imwrite(os.path.join(targetFolder, maskName), tileImg)

    



