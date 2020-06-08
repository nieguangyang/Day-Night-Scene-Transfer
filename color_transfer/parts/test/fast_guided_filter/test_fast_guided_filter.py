import os
import numpy as np
import matplotlib.pyplot as plt

from color_transfer.parts.pilutil import imread, imresize
from color_transfer.parts.fast_guided_filter import fast_guided_filter

path = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
a = imread(path + "/a.jpg")
b = imread(path + "/b.jpg")
w, h = a.shape[:2]
w, h = w // 8, h // 8
a_low = imresize(a, (w, h)) / 255
b_low = imresize(b, (w, h)) / 255
a_high = a / 255
# equal channels
b_high = fast_guided_filter(a_low, b_low, a_high)
plt.subplot(221).imshow(a_low)
plt.subplot(222).imshow(a_high)
plt.subplot(223).imshow(b_low)
plt.subplot(224).imshow(b_high)
plt.show()
# different channels
a_low = np.mean(a_low, axis=2, keepdims=True)
a_high = np.mean(a_high, axis=2, keepdims=True)
b_high = fast_guided_filter(a_low, b_low, a_high)
plt.subplot(221).imshow(a_low[:, :, 0])
plt.subplot(222).imshow(a_high[:, :, 0])
plt.subplot(223).imshow(b_low)
plt.subplot(224).imshow(b_high)
plt.show()
