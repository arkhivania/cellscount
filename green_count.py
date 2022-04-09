import argparse
import imageio
from skimage import measure

parser = argparse.ArgumentParser(description='Count green dots in image')
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

image = imageio.imread(args.input)
if image.shape[-1] == 4:
    image = image[:,:,(0,1,2)]

mask = (image == [0,255,0]).all(-1) * 255
labels = measure.label(mask)
print("Count of green dots: {}".format(labels.max()))
