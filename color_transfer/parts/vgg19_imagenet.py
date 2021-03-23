import os
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU
from keras.models import Model

from color_transfer.parts.pilutil import preprocess
from color_transfer.weights import VGG19_IMAGENET_WEIGHTS, PATH


def vgg_block(filters, conv_layers, block_id):
    """
    :param filters: number of filters
    :param conv_layers: number of conv2d layer in the block
    :param block_id: block index
    :return block:
    """
    def block(x):
        _ = x
        for conv_id in range(1, conv_layers + 1):
            _ = Conv2D(filters, 3, padding="same", name="conv%d_%d" % (block_id, conv_id))(_)
            _ = ReLU(name="relu%d_%d" % (block_id, conv_id))(_)
        y = MaxPooling2D(2, name="pool%d" % block_id)(_)
        return y
    return block


def build_vgg(input_shape, block_filters, block_layers, weights=None):
    """
    :param input_shape: (height, width, 3)
        (224, 224, 3) for pre-trained VGG16 or VGG19
    :param block_filters: [filters1, filters2, ...]
        [64, 128, 256, 512, 512] for VGG16 or VGG19
    :param block_layers: [conv_layers1, conv_layers2, ...]
        [2, 2, 3, 3, 3] for VGG16
        [2, 2, 4, 4, 4] for VGG19
    :param weights: full path to weights file
    :return model:
    """
    _ = x = Input(input_shape, name="input")
    for i, (filters, conv_layers) in enumerate(zip(block_filters, block_layers)):
        _ = vgg_block(filters, conv_layers, block_id=i + 1)(_)
    y = _
    model = Model(x, y)
    if weights is not None:
        model.load_weights(weights)
    return model


def vgg19(weights=None):
    """
    :param weights: full path to weights file
    """
    input_shape = (224, 224, 3)
    block_filters = [64, 128, 256, 512, 512]
    block_layers = [2, 2, 4, 4, 4]
    return build_vgg(input_shape, block_filters, block_layers, weights)


def vgg19_imagenet():
    """
    VGG19 pre-trained on ImageNet.
    """
    weights = VGG19_IMAGENET_WEIGHTS
    if not os.path.exists(weights):
        filename = weights.split("/")[-1]
        link = (
            "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/"
            "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
        instr = (
            "Please download weight file %s from %s,\n" % (filename, link),
            "then put it under directory %s" % PATH
        )
        raise Exception(instr)
    return vgg19(weights)


def rescale(img, eps=1e-4):
    """
    min-max normalization
    """
    height, width, channels = img.shape
    flattened = img.reshape((-1, channels))
    _min = flattened.min(axis=0, keepdims=True).reshape((1, 1, channels))
    _max = flattened.max(axis=0, keepdims=True).reshape((1, 1, channels))
    normalized = (img - _min) / (_max - _min + eps)
    return normalized


def standardize(img, eps=1e-4):
    """
    instance normalization
    """
    height, width, channels = img.shape
    flattened = img.reshape((-1, channels))
    mean = flattened.mean(axis=0, keepdims=True).reshape((1, 1, channels))
    var = flattened.var(axis=0, keepdims=True).reshape((1, 1, channels))
    normalized = (img - mean) / np.sqrt(var + eps)
    return normalized


class FeatureFromVGG19ImageNet:
    """
    Extract feature map from VGG pre-trained on ImageNet.
    """

    def __init__(self, interp=2):
        """
        :param interp: 0 - nearest, 1 - lanczos, 2 - bilinear, 3 - cubic
        """
        model = vgg19_imagenet()
        self.size = model.input_shape[1:-1]
        self.interp = interp
        # models to extract feature map at different level
        self.models = [
            Model(model.input, model.get_layer("relu1_1").output),  # 1
            Model(model.input, model.get_layer("relu2_1").output),  # 2
            Model(model.input, model.get_layer("relu3_1").output),  # 3
            Model(model.input, model.get_layer("relu4_1").output),  # 4
            Model(model.input, model.get_layer("relu5_1").output)   # 5
        ]
        self.output_shapes = [model.output_shape[1:] for model in self.models]

    def get_model(self, level):
        """
        :param level: level from 1 to 5
        """
        return self.models[level - 1]

    def get_output_shape(self, level):
        """
        Get feature map shape at given level
        :param level: level from 1 to 5
        """
        return self.output_shapes[level - 1]

    def extract(self, img, level, normalize=None):
        """
        :param img: (height, width, 3) ndarray of RGB image
        :param level: level from 1 to 5
        :param normalize: None / "rescale" / "standardize", how to normalize each channel of feature map
            None, no normalization
            rescale, (x - x_min) / (x_max - x_min)
            standardize, (x - x_mean) / x_var
        :return feature: feature map at given level w.r.t. given image
        """
        x = preprocess(img, self.size, self.interp)
        model = self.get_model(level)
        feature = model.predict(x)[0]  # feature map
        if normalize == "rescale":
            feature = rescale(feature)
        elif normalize == "standardize":
            feature = standardize(feature)
        return feature
