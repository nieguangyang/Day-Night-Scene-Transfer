import os
import matplotlib.pyplot as plt

from color_transfer.parts.pspnet50_ade20k import PSPNet50ADE20K
from color_transfer.parts.pilutil import imread


path = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
pattern = path + "/%d.jpg"
weights = "E:/ai/weights/pretrained/pspnet50_ade20k.h5"
psp = PSPNet50ADE20K(weights)
for i in range(10):
    file = pattern % i
    img = imread(file)
    predicted = psp.colorize(psp.predict(img))
    ax = plt.subplot(121)
    ax.imshow(img)
    ax = plt.subplot(122)
    ax.imshow(predicted)
    plt.show()



