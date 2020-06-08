"""
Inspired by <<Progressive Color Transfer with Dense Semantic Correspondences>> (https://arxiv.org/abs/1710.00756)
and PyTorch implementation by rassilon712 (https://github.com/rassilon712/Neural_Color_Transfer).

Differences from the original:
    1. use instance normalization per feature map from VGG19
    2. cluster on pair feature maps of source and reference
    3. only perform color transferring at level 1
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time

from color_transfer.parts.pilutil import imread, imresize
from color_transfer.parts.vgg19_imagenet import FeatureFromVGG19ImageNet
from color_transfer.parts.fast_guided_filter import fast_guided_filter
from color_transfer.parts.nnf_computation import nn_search as nn_search_py, bds_vote as bds_vote_py
from color_transfer.parts.nnf_computation_c import nn_search as nn_search_c, bds_vote as bds_vote_c
from color_transfer.parts.transfer_estimate import transfer_estimate


def cluster_on_feature_pair(f1, f2, n_clusters):
    """
    :param f1: H*W*C ndarray, feature map of image 1
    :param f2: H*W*C ndarray, feature map of image 2
    :param n_clusters: number of clusters
    :return label1: H*W ndarray, label index for each pixel of image 1
    :return label2: H*W ndarray, label index for each pixel of image 2
    :return one_hot1: H*W ndarray, one-hot for each pixel of image 1
    :return one_hot2: H*W ndarray, one-hot for each pixel of image 2
    """
    assert f1.shape == f2.shape
    h, w, c = f1.shape
    f = np.concatenate([f1, f2], axis=1)
    km = KMeans(n_clusters=n_clusters).fit(f.reshape((-1, c)))
    label = km.labels_.reshape((h, 2 * w))
    one_hot = (np.arange(n_clusters) == label[:, :, None]).astype(float)
    label1, label2 = label[:, :w], label[:, w:]
    one_hot1, one_hot2 = one_hot[:, :w], one_hot[:, w:]
    return label1, label2, one_hot1, one_hot2


def display(images, rows=1, columns=None):
    if columns is None:
        columns = len(images)
    for i, img in enumerate(images):
        ax = plt.subplot(rows, columns, i + 1)
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


class ClusterTransfer:
    def __init__(self, weights, patch_match_c=True):
        """
        :param weights: full path to weights file (download from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
        :param patch_match_c: whether use C-implemented patch match, which is far more efficient than the python one
        """
        self.vgg19 = FeatureFromVGG19ImageNet(weights)
        if patch_match_c:
            self.nn_search = nn_search_c
            self.bds_vote = bds_vote_c
        else:
            self.nn_search = nn_search_py
            self.bds_vote = bds_vote_py

    def transfer(self, src, ref, level=1, n_clusters=3, interp=2, normalize="standardize", total_iter=5, epochs=250):
        """
        :param src: H*W*3 ndarray, source image
        :param ref: H*W*3 ndarray, reference image
        :param level: 1 to 5
        :param n_clusters: number of clusters, it depends on how many elements in the image
        :param interp: 0 - nearest, 1 - lanczos, 2 - bilinear, 3 - cubic
        :param normalize: normalization per channel, "rescale" / "standardize"
        :param total_iter:
        :param epochs:
        :return transferred: color-transferred image
        """
        start_time = time()
        size = self.vgg19.get_output_shape(level)[:2]
        s = imresize(src, size, interp) / 255
        r = imresize(ref, size, interp) / 255
        print("extract feature maps")
        fs = self.vgg19.extract(src, level, normalize)
        fr = self.vgg19.extract(ref, level, normalize)
        print("cluster")
        lbs, lbr, ohs, ohr = cluster_on_feature_pair(fs, fr, n_clusters)  # lb -> label, oh -> one-hot
        # display([s, lbs, lbr, r])
        print("concatenate")
        fs = np.concatenate([ohs, fs], axis=2)
        fr = np.concatenate([ohr, fr], axis=2)
        print("patchmatch")
        nnf_sr, nnf_rs = self.nn_search(fs, fr, total_iter=total_iter)
        g = self.bds_vote(r, nnf_sr, nnf_rs)
        fg = self.bds_vote(fr, nnf_sr, nnf_rs)
        print("transfer estimate")
        t, a, b = transfer_estimate(s, g, fs, fg, lambda_d=100, lambda_l=0.01, epochs=epochs)
        a = fast_guided_filter(s, a, src / 255)
        b = fast_guided_filter(s, b, src / 255)
        transferred = a * src + b * 255
        transferred[transferred > 255] = 255
        transferred = transferred.astype(np.uint8)
        time_cost = time() - start_time
        print("time cost: %.2fs" % time_cost)
        return transferred


def test():
    weights = "D:/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

    from color_transfer.img import PATH
    src_file, ref_file = PATH + "/day.jpg", PATH + "/night.jpg"

    src = imread(src_file)
    ref = imread(ref_file)

    patch_match_c = True
    # patch_match_c = False
    interp = 2  # bilinear
    normalize = "standardize"
    level = 1
    n_clusters = 3  # 3 for sky, building and ground
    total_iter = 5  # number of iterations for PatchMatch
    epochs = 250  # number of epochs for transfer estimate

    ct = ClusterTransfer(weights, patch_match_c)
    transferred = ct.transfer(src, ref, level, n_clusters, interp, normalize, total_iter, epochs)
    display([src, ref, transferred])


if __name__ == "__main__":
    test()
