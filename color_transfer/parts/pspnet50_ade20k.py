"""
PSPNet50 pre-trained on ADE20K.
https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow
"""
import os
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import ZeroPadding2D, Add, MaxPooling2D
from keras.layers import Layer, AveragePooling2D, Concatenate
from keras.layers import Input, Dropout
from keras.backend import tf as ktf
from keras.models import Model
from keras.optimizers import SGD

from color_transfer.parts.pilutil import preprocess


def conv2d(n_filters, kernel_size, strides=1, padding="valid", dilation_rate=1, **args):
    return Conv2D(n_filters, kernel_size,
                  strides=strides, padding=padding, dilation_rate=dilation_rate, use_bias=False,
                  **args)


def bn():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)


def relu():
    return Activation("relu")


def conv_block(n_filters, strides, padding):
    """
    A residual block that has a conv layer at shortcut.
    :param n_filters: number of filters at last conv layer
    :param strides: step size
    :param padding: padding width
    :return block:
    """
    def block(x):
        _ = relu()(x)
        # main branch
        m = conv2d(n_filters // 4, 1, strides=strides)(_)  # 1st conv layer
        m = bn()(m)
        m = relu()(m)
        m = ZeroPadding2D(padding)(m)
        m = conv2d(n_filters // 4, 3, dilation_rate=padding)(m)  # 2nd conv layer
        m = bn()(m)
        m = relu()(m)
        m = conv2d(n_filters, 1)(m)  # 3rd conv layer
        m = bn()(m)
        # shortcut branch
        s = conv2d(n_filters, 1, strides=strides)(_)
        s = bn()(s)
        # add
        y = Add()([m, s])
        return y
    return block


def id_block(n_filters, padding):  # residual
    """
    A residual block that has no conv layer at shortcut.
    :param n_filters: number of filters at last conv layer
    :param padding: padding width
    :return block:
    """
    def block(x):
        _ = relu()(x)
        # main branch
        m = conv2d(n_filters // 4, 1)(_)  # 1st conv layer
        m = bn()(m)
        m = relu()(m)
        m = ZeroPadding2D(padding)(m)
        m = conv2d(n_filters // 4, 3, dilation_rate=padding)(m)  # 2nd conv layer
        m = bn()(m)
        m = relu()(m)
        m = conv2d(n_filters, 1)(m)  # 3rd conv layer
        m = bn()(m)
        # shortcut branch
        s = _
        # add
        y = Add()([m, s])
        return y
    return block


def res_block(n_filters, strides, padding, n_layers):
    """
    :param n_filters: number of filters at last conv layer
    :param strides: step size
    :param padding: padding width
    :param n_layers: number of weight layers
    :return block:
    """
    def block(x):
        _ = conv_block(n_filters, strides, padding)(x)
        for i in range(n_layers - 1):
            _ = id_block(n_filters, padding)(_)
        y = _
        return y
    return block


def resnet(n_layers):
    """
    :param n_layers: number of weight layers in resnet
    :return block:
    """
    def block(x):
        _ = conv2d(64, 3, strides=2, padding="same")(x)
        _ = bn()(_)
        _ = relu()(_)
        _ = conv2d(64, 3, padding="same")(_)
        _ = bn()(_)
        _ = relu()(_)
        _ = conv2d(128, 3, padding="same")(_)
        _ = bn()(_)
        _ = relu()(_)
        _ = MaxPooling2D(pool_size=3, padding='same', strides=2)(_)
        # res blocks
        _ = res_block(256, strides=1, padding=1, n_layers=3)(_)
        _ = res_block(512, strides=2, padding=1, n_layers=4)(_)
        if n_layers == 50:
            _ = res_block(1024, strides=1, padding=2, n_layers=6)(_)
        elif n_layers == 101:
            _ = res_block(1024, strides=1, padding=2, n_layers=23)(_)
        else:
            raise Exception("n_layers should be 50 or 101")
        _ = res_block(2048, strides=1, padding=4, n_layers=3)(_)
        y = relu()(_)
        return y
    return block


class Interp(Layer):
    def __init__(self, target_size, **kwargs):
        """
        :param target_size: (target_height, target_width)
        :param kwargs:
        """
        self.target_size = target_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        resized = ktf.image.resize_images(inputs, self.target_size, align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        height, width = self.target_size
        channels = input_shape[3]
        output_shape = (None, height, width, channels)
        return output_shape

    def get_config(self):
        config = super(Interp, self).get_config()
        config['target_size'] = self.target_size
        return config


def pooling_block(feature_map_size, bin_size):
    """
    :param feature_map_size: (height, width) for feature map tensor
    :param bin_size: bin size of pyramid pooling, should be 1 / 2 / 3 / 6
    :return block:
    """
    h, w = feature_map_size
    if h != w:
        raise Exception("height and width of feature map should be identical")
    if h % 6 != 0:
        raise Exception("height and width of feature map should be multiples of 6")
    pool_size = h // bin_size
    strides = pool_size
    target_size = (h, h)

    def block(x):
        _ = AveragePooling2D(pool_size, strides=strides)(x)
        _ = conv2d(512, 1)(_)
        _ = bn()(_)
        _ = relu()(_)
        y = Interp(target_size)(_)
        return y
    return block


def pyramid_pooling(feature_map_size):
    """
    :param feature_map_size: (height, width) for feature map tensor
    :return block:
    """
    def block(x):
        bin1 = pooling_block(feature_map_size, 1)(x)
        bin2 = pooling_block(feature_map_size, 2)(x)
        bin3 = pooling_block(feature_map_size, 3)(x)
        bin6 = pooling_block(feature_map_size, 6)(x)
        y = Concatenate()([x, bin6, bin3, bin2, bin1])
        return y
    return block


def build_pspnet(input_shape, n_resnet_layers, n_classes):
    """
    :param input_shape: (height, width), (473, 473) and (713, 713) are supported
    :param n_resnet_layers: number of resnet layers
    :param n_classes: number of classes
    :return model: keras model
    """
    if input_shape == (473, 473):
        feature_map_size = (60, 60)
    elif input_shape == (713, 713):
        feature_map_size = (90, 90)
    else:
        raise Exception("For now, only (473, 473) and (713, 713) are supported.")
    h, w = input_shape
    x = Input((h, w, 3))
    _ = resnet(n_resnet_layers)(x)
    _ = pyramid_pooling(feature_map_size)(_)
    _ = conv2d(512, 3, padding="same")(_)
    _ = bn()(_)
    _ = Dropout(0.1)(_)
    _ = Conv2D(n_classes, 1)(_)  # use bias
    _ = Interp((h, w))(_)
    y = Activation("softmax")(_)
    model = Model(x, y)
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def pspnet50_ade20k(weights=None):
    """
    PSPNet pre-trained on ADE20K.
    :param weights: full path to pre-trained weights file
    :return:
    """
    if weights is None:
        path = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
        weights = path + "/pspnet50_ade20k.h5"
        if not os.path.exists(weights):
            link = "https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5"
            instr = "\n".join((
                "Please down weight file from %s " % link,
                "and put it under directory %s" % path
            ))
            raise Exception(instr)
    input_shape = (473, 473)
    n_resnet_layers = 50
    n_classes = 150
    model = build_pspnet(input_shape, n_resnet_layers, n_classes)
    model.load_weights(weights)
    return model


def resize(img, size):
    """
    Resize for image whose channels are not 3.
    :param img: (height, width, channels) ndarray
    :param size: (target_height, target_width)
    :return resized: resized image
    """
    height, width, channels = img.shape
    target_height, target_width = size
    resized = np.zeros((target_height, target_width, channels), dtype=img.dtype)
    yr = (height - 1) / (target_height - 1)
    xr = (width - 1) / (target_width - 1)
    for ty in range(target_height):
        for tx in range(target_width):
            y = round(yr * ty)
            x = round(xr * tx)
            resized[ty, tx] = img[y, x]
    return resized


labels = [
    dict(label="wall", color=(120, 120, 120)),
    dict(label="building", color=(180, 120, 120)),
    dict(label="sky", color=(6, 230, 230)),
    dict(label="floor", color=(80, 50, 50)),
    dict(label="tree", color=(4, 200, 3)),
    dict(label="ceiling", color=(120, 120, 80)),
    dict(label="road", color=(140, 140, 140)),
    dict(label="bed", color=(204, 5, 255)),
    dict(label="windowpane", color=(230, 230, 230)),
    dict(label="grass", color=(4, 250, 7)),
    dict(label="cabinet", color=(224, 5, 255)),
    dict(label="sidewalk", color=(235, 255, 7)),
    dict(label="person", color=(150, 5, 61)),
    dict(label="earth", color=(120, 120, 70)),
    dict(label="door", color=(8, 255, 51)),
    dict(label="table", color=(255, 6, 82)),
    dict(label="mountain", color=(143, 255, 140)),
    dict(label="plant", color=(204, 255, 4)),
    dict(label="curtain", color=(255, 51, 7)),
    dict(label="chair", color=(204, 70, 3)),
    dict(label="car", color=(0, 102, 200)),
    dict(label="water", color=(61, 230, 250)),
    dict(label="painting", color=(255, 6, 51)),
    dict(label="sofa", color=(11, 102, 255)),
    dict(label="shelf", color=(255, 7, 71)),
    dict(label="house", color=(255, 9, 224)),
    dict(label="sea", color=(9, 7, 230)),
    dict(label="mirror", color=(220, 220, 220)),
    dict(label="rug", color=(255, 9, 92)),
    dict(label="field", color=(112, 9, 255)),
    dict(label="armchair", color=(8, 255, 214)),
    dict(label="seat", color=(7, 255, 224)),
    dict(label="fence", color=(255, 184, 6)),
    dict(label="desk", color=(10, 255, 71)),
    dict(label="rock", color=(255, 41, 10)),
    dict(label="wardrobe", color=(7, 255, 255)),
    dict(label="lamp", color=(224, 255, 8)),
    dict(label="bathtub", color=(102, 8, 255)),
    dict(label="railing", color=(255, 61, 6)),
    dict(label="cushion", color=(255, 194, 7)),
    dict(label="base", color=(255, 122, 8)),
    dict(label="box", color=(0, 255, 20)),
    dict(label="column", color=(255, 8, 41)),
    dict(label="signboard", color=(255, 5, 153)),
    dict(label="chest of drawers", color=(6, 51, 255)),
    dict(label="counter", color=(235, 12, 255)),
    dict(label="sand", color=(160, 150, 20)),
    dict(label="sink", color=(0, 163, 255)),
    dict(label="skyscraper", color=(140, 140, 140)),
    dict(label="fireplace", color=(250, 10, 15)),
    dict(label="refrigerator", color=(20, 255, 0)),
    dict(label="grandstand", color=(31, 255, 0)),
    dict(label="path", color=(255, 31, 0)),
    dict(label="stairs", color=(255, 224, 0)),
    dict(label="runway", color=(153, 255, 0)),
    dict(label="case", color=(0, 0, 255)),
    dict(label="pool table", color=(255, 71, 0)),
    dict(label="pillow", color=(0, 235, 255)),
    dict(label="screen door", color=(0, 173, 255)),
    dict(label="stairway", color=(31, 0, 255)),
    dict(label="river", color=(11, 200, 200)),
    dict(label="bridge", color=(255, 82, 0)),
    dict(label="bookcase", color=(0, 255, 245)),
    dict(label="blind", color=(0, 61, 255)),
    dict(label="coffee table", color=(0, 255, 112)),
    dict(label="toilet", color=(0, 255, 133)),
    dict(label="flower", color=(255, 0, 0)),
    dict(label="book", color=(255, 163, 0)),
    dict(label="hill", color=(255, 102, 0)),
    dict(label="bench", color=(194, 255, 0)),
    dict(label="countertop", color=(0, 143, 255)),
    dict(label="stove", color=(51, 255, 0)),
    dict(label="palm", color=(0, 82, 255)),
    dict(label="kitchen island", color=(0, 255, 41)),
    dict(label="computer", color=(0, 255, 173)),
    dict(label="swivel chair", color=(10, 0, 255)),
    dict(label="boat", color=(173, 255, 0)),
    dict(label="bar", color=(0, 255, 153)),
    dict(label="arcade machine", color=(255, 92, 0)),
    dict(label="hovel", color=(255, 0, 255)),
    dict(label="bus", color=(255, 0, 245)),
    dict(label="towel", color=(255, 0, 102)),
    dict(label="light", color=(255, 173, 0)),
    dict(label="truck", color=(255, 0, 20)),
    dict(label="tower", color=(255, 184, 184)),
    dict(label="chandelier", color=(0, 31, 255)),
    dict(label="awning", color=(0, 255, 61)),
    dict(label="streetlight", color=(0, 71, 255)),
    dict(label="booth", color=(255, 0, 204)),
    dict(label="television", color=(0, 255, 194)),
    dict(label="airplane", color=(0, 255, 82)),
    dict(label="dirt track", color=(0, 10, 255)),
    dict(label="apparel", color=(0, 112, 255)),
    dict(label="pole", color=(51, 0, 255)),
    dict(label="land", color=(0, 194, 255)),
    dict(label="bannister", color=(0, 122, 255)),
    dict(label="escalator", color=(0, 255, 163)),
    dict(label="ottoman", color=(255, 153, 0)),
    dict(label="bottle", color=(0, 255, 10)),
    dict(label="buffet", color=(255, 112, 0)),
    dict(label="poster", color=(143, 255, 0)),
    dict(label="stage", color=(82, 0, 255)),
    dict(label="van", color=(163, 255, 0)),
    dict(label="ship", color=(255, 235, 0)),
    dict(label="fountain", color=(8, 184, 170)),
    dict(label="conveyer belt", color=(133, 0, 255)),
    dict(label="canopy", color=(0, 255, 92)),
    dict(label="washer", color=(184, 0, 255)),
    dict(label="plaything", color=(255, 0, 31)),
    dict(label="swimming pool", color=(0, 184, 255)),
    dict(label="stool", color=(0, 214, 255)),
    dict(label="barrel", color=(255, 0, 112)),
    dict(label="basket", color=(92, 255, 0)),
    dict(label="waterfall", color=(0, 224, 255)),
    dict(label="tent", color=(112, 224, 255)),
    dict(label="bag", color=(70, 184, 160)),
    dict(label="minibike", color=(163, 0, 255)),
    dict(label="cradle", color=(153, 0, 255)),
    dict(label="oven", color=(71, 255, 0)),
    dict(label="ball", color=(255, 0, 163)),
    dict(label="food", color=(255, 204, 0)),
    dict(label="step", color=(255, 0, 143)),
    dict(label="tank", color=(0, 255, 235)),
    dict(label="trade name", color=(133, 255, 0)),
    dict(label="microwave", color=(255, 0, 235)),
    dict(label="pot", color=(245, 0, 255)),
    dict(label="animal", color=(255, 0, 122)),
    dict(label="bicycle", color=(255, 245, 0)),
    dict(label="lake", color=(10, 190, 212)),
    dict(label="dishwasher", color=(214, 255, 0)),
    dict(label="screen", color=(0, 204, 255)),
    dict(label="blanket", color=(20, 0, 255)),
    dict(label="sculpture", color=(255, 255, 0)),
    dict(label="hood", color=(0, 153, 255)),
    dict(label="sconce", color=(0, 41, 255)),
    dict(label="vase", color=(0, 255, 204)),
    dict(label="traffic light", color=(41, 0, 255)),
    dict(label="tray", color=(41, 255, 0)),
    dict(label="ashcan", color=(173, 0, 255)),
    dict(label="fan", color=(0, 245, 255)),
    dict(label="pier", color=(71, 0, 255)),
    dict(label="crt screen", color=(122, 0, 255)),
    dict(label="plate", color=(0, 255, 184)),
    dict(label="monitor", color=(0, 92, 255)),
    dict(label="bulletin board", color=(184, 255, 0)),
    dict(label="shower", color=(0, 133, 255)),
    dict(label="radiator", color=(255, 214, 0)),
    dict(label="glass", color=(25, 194, 194)),
    dict(label="clock", color=(102, 255, 0)),
    dict(label="flag", color=(92, 0, 255))
]


class PSPNet50ADE20K:
    """
    PSPNet50 pre-trained on ADE20K.
    """
    def __init__(self, weights=None):
        """
        :param weights: full path to pre-trained weights file
        """
        self.model = pspnet50_ade20k(weights)
        self.size = self.model.input_shape[1:-1]

    def predict(self, img):
        """
        :param img: (height, width, channels) ndarray
        :return label: (height, width, channels) ndarray, one-hot label for each pixel
        """
        x = preprocess(img, self.size)
        y = self.model.predict(x)
        label = resize(y[0], img.shape[:2])
        return label

    @staticmethod
    def colorize(label):
        index = np.argmax(label, axis=2)
        h, w = index.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                i = index[y, x]
                if i >= 150:
                    colored[y, x, :] = (0, 0, 0)
                else:
                    colored[y, x, :] = labels[i]["color"]
        return colored
