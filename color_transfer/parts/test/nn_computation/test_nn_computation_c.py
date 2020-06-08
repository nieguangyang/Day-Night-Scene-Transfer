import os
import numpy as np
import matplotlib.pyplot as plt
from color_transfer.parts.pilutil import imread

from color_transfer.parts.nnf_computation_c import nn_search, bds_vote

path = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
src = imread(path + "/a.jpg")
ref = imread(path + "/a.jpg")
patch_size = 3
total_iter = 2
w = 1

nnf_sr, nnf_rs = nn_search(src, ref, patch_size, total_iter)  # nn search
guide = bds_vote(ref, nnf_sr, nnf_rs, patch_size, w)  # bds vote
plt.subplot(121).imshow(src)
plt.subplot(122).imshow(guide.astype(np.uint8))
plt.show()
