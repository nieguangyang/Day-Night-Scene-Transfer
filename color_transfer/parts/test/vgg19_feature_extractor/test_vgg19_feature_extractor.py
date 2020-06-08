import matplotlib.pyplot as plt
from scipy.misc import imread

from color_transfer.parts.vgg19_imagenet import FeatureFromVGG19ImageNet


image = "./test.jpg"  # path to image
weights = "E:/ai/weights/pretrained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"  # path to weights file
levels = 5  # max 5
channels = 16  # how many feature maps displayed for each level
normalize = "standardize"

img = imread(image)
v = FeatureFromVGG19ImageNet(weights)
for level in range(1, levels + 1):
    f = v.extract(img, level, normalize)
    for c in range(channels):
        ax = plt.subplot(levels, channels, (level - 1) * channels + c + 1)
        ax.imshow(f[:, :, c])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
plt.show()
