import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import tifffile as tiff
import os
from PIL import Image
import time


root_path = "/media/huifang/data/sennet/xenium/"

dataset = open(root_path+'data_list.txt')
lines = dataset.readlines()
for i in range(len(lines)):
    print(i)
    line = lines[i].rstrip().split(' ')
    subfolder = os.path.join(root_path,line[0])
    print(subfolder)
    if os.path.exists(os.path.join(subfolder,'outs')):
        subfolder = os.path.join(subfolder,'outs')

    ome_tiff_path = subfolder+'/morphology_focus/morphology_focus_0000.ome.tif'

    image_data = tiff.imread(ome_tiff_path)
    channel_img = image_data[0,:,:]

    # Convert to 8-bit with contrast stretching
    if channel_img.dtype != 'uint8':
        p1, p99 = np.percentile(channel_img, [1, 99])
        channel_img = np.clip(channel_img, p1, p99)
        channel_img = (255 * (channel_img - p1) / (p99 - p1 + 1e-8)).astype('uint8')
    # Convert to PIL Image
    im = Image.fromarray(channel_img)

    out_dir = subfolder+'/morphology_focus/'
    # Resize to quarter size
    im_quarter = im.resize((im.width // 4, im.height // 4), Image.LANCZOS)
    # plt.imshow(im_quarter)
    # plt.show()
    im_one_sixteenth = im.resize((im.width // 16, im.height // 16), Image.LANCZOS)

    im_quarter.save(os.path.join(out_dir, f"channel0_quarter.png"))
    im_one_sixteenth.save(os.path.join(out_dir, f"channel0_one_sixteenth.png"))
    print('saved')
    # test = input()







