import numpy as np
import matplotlib.pyplot as plt
from time import time

from color_transfer.parts.pilutil import imread, imresize
from color_transfer.parts.vgg19_imagenet import FeatureFromVGG19ImageNet
from color_transfer.parts.pspnet50_ade20k import PSPNet50ADE20K
from color_transfer.parts.fast_guided_filter import fast_guided_filter
from color_transfer.parts.nnf_computation import nn_search as nn_search_py, bds_vote as bds_vote_py
from color_transfer.parts.nnf_computation_c import nn_search as nn_search_c, bds_vote as bds_vote_c
from color_transfer.parts.transfer_estimate import transfer_estimate


class ADE20KTransfer:
    def __init__(self, vgg19_imagenet_weights, pspnet50_ade20k_weights, patch_match_c=True):
        """
        :param vgg19_imagenet_weights: full path to weights of VGG19 pre-trained on ImageNet (download from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
        :param pspnet50_ade20k_weights: full path to weights of PSPNet50 pre-trained on (download from https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5)
        :param patch_match_c: whether use C-implemented patch match, which is far more efficient than the python one
        """
        self.vgg19 = FeatureFromVGG19ImageNet(vgg19_imagenet_weights)
        self.psp50 = PSPNet50ADE20K(pspnet50_ade20k_weights)
        if patch_match_c:
            self.nn_search = nn_search_c
            self.bds_vote = bds_vote_c
        else:
            self.nn_search = nn_search_py
            self.bds_vote = bds_vote_py

    def transfer(self, src, ref, level=1, interp=2, normalize="standardize"):
        """
        :param src: H*W*3 ndarray, source image
        :param ref: H*W*3 ndarray, reference image
        :param level: 1 to 5
        :param interp: 0 - nearest, 1 - lanczos, 2 - bilinear, 3 - cubic
        :param normalize: normalization per channel, "rescale" / "standardize"
        :return transferred: color-transferred image
        """
        start_time = time()
        size = self.vgg19.get_output_shape(level)[:2]
        # resized and rescaled
        s = imresize(src, size, interp) / 255
        r = imresize(ref, size, interp) / 255
        # feature map from VGG19 on ImageNet
        fs = self.vgg19.extract(src, level, normalize)
        fr = self.vgg19.extract(ref, level, normalize)
        # label from PSPNet on ADE20K
        lbs = self.psp50.predict(s * 255)
        lbr = self.psp50.predict(r * 255)
        # concat feature map and label
        fs = np.concatenate([lbs, fs], axis=2)
        fr = np.concatenate([lbr, fr], axis=2)
        nnf_sr, nnf_rs = self.nn_search(fs, fr)
        g = self.bds_vote(r, nnf_sr, nnf_rs)  # r size wrong
        fg = self.bds_vote(fr, nnf_sr, nnf_rs)
        t, a, b = transfer_estimate(s, g, fs, fg, lambda_d=100, lambda_l=0.01)
        a = fast_guided_filter(s, a, src / 255)
        b = fast_guided_filter(s, b, src / 255)
        transferred = a * src + b * 255
        transferred[transferred > 255] = 255
        transferred = transferred.astype(np.uint8)
        time_cost = time() - start_time
        print("time cost: %.2fs" % time_cost)
        return transferred


def display(images, rows=1, columns=None):
    if columns is None:
        columns = len(images)
    for i, img in enumerate(images):
        ax = plt.subplot(rows, columns, i + 1)
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


def test():
    vgg19_imagenet_weights = "E:/ai/weights/pretrained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pspnet50_ade20k_weights = "E:/ai/weights/pretrained/pspnet50_ade20k.h5"

    from color_transfer.img import PATH
    src_file, ref_file = PATH + "/day.jpg", PATH + "/night.jpg"

    src = imread(src_file)
    ref = imread(ref_file)

    patch_match_c = True
    # patch_match_c = False
    interp = 2  # bilinear
    normalize = "standardize"
    level = 1

    t = ADE20KTransfer(vgg19_imagenet_weights, pspnet50_ade20k_weights, patch_match_c)
    transferred = t.transfer(src, ref, level, interp, normalize)
    display([src, ref, transferred])


if __name__ == "__main__":
    test()
