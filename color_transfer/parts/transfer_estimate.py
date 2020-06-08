"""
https://github.com/rassilon712/Neural_Color_Transfer/blob/master/models.py
"""

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from keras.layers import Layer
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


class TransferLayer(Layer):
    def __init__(self, init_a=None, init_b=None, **kwargs):
        """
        :param init_a: (height, weights, height) ndarray, initialized a
        :param init_b: (height, weights, height) ndarray, initialized b
        :param kwargs:
        """
        self.init_a, self.init_b = init_a, init_b
        self.a, self.b = None, None
        self.shape = None
        super(TransferLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, height, width, channels = input_shape
        self.shape = (height, width, channels)
        if self.init_a is None:
            init_a = np.random.rand(*self.shape)
        else:
            assert self.init_a.shape == self.shape
            init_a = self.init_a
        if self.init_b is None:
            init_b = np.random.rand(*self.shape)
        else:
            assert self.init_b.shape == self.shape
            init_b = self.init_b
        self.a = K.variable(init_a, dtype=K.floatx(), name="a")
        self.b = K.variable(init_b, dtype=K.floatx(), name="b")
        self._trainable_weights.extend([self.a, self.b])
        super(TransferLayer, self).build(input_shape)

    def call(self, inputs, *_):
        s = inputs
        t = self.a * s + self.b  # transferred src
        return [t, self.a, self.b]

    def compute_output_shape(self, input_shape):
        return [input_shape, self.shape, self.shape]


def calc_confidence(fs, fg):
    """
    :param fs: (height, width, channels) ndarray, feature map of source from VGG19
    :param fg: (height, width, channels) ndarray, feature map of guide from VGG19
    :return confidence: (height, weight) ndarray, normalized weight to give high confidence to well-matched points
    """
    me = np.sum(np.square(fs - fg), axis=2)
    confidence = 1 - me / np.linalg.norm(me, ord=2)
    return confidence


class LossD(Layer):
    """
    loss for pixel-wise matching difference
    """
    def __init__(self, confidence, **kwargs):
        """
        :param level: level from 1 to 5
        :param confidence:(height, weight) ndarray, normalized weight to give high confidence to well-matched points
        :param kwargs:
        """
        self.confidence = K.constant(confidence)
        super(LossD, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LossD, self).build(input_shape)

    def call(self, inputs, *_):
        """
        :param inputs: (t, g)
            t: transferred image of source
            g: image of guide
        """
        t, g = inputs
        error = K.sum(K.square(t - g), axis=-1)
        loss = K.mean(K.batch_flatten(self.confidence * error), axis=1, keepdims=True)
        return loss

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], 1)
        return output_shape


def calc_luminance(img):
    """
    :param img: (height, width, 3) ndarray, RGB image
    """
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance


class LossL(Layer):
    """
    loss for local constraint
    https://stackoverflow.com/questions/48215077/how-to-shift-values-in-tensor?rq=1
    """
    def __init__(self, luminance, **kwargs):
        """
        :param luminance: luminance of source
        :param kwargs:
        """
        self.luminance = K.constant(luminance)
        super(LossL, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LossL, self).build(input_shape)

    @staticmethod
    def shift(x):
        """
        shift tensor of image on 4 directions up, down, left, right
        :param x: (height, width, channels) or (height, width) tensor
        """
        dims = len(x.shape)
        if dims == 3:
            up = K.concatenate([x[1:, :, :], x[-1:, :, :]], axis=0)
            down = K.concatenate([x[:1, :, :], x[:-1, :, :]], axis=0)
            left = K.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
            right = K.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
        elif dims == 2:
            up = K.concatenate([x[1:, :], x[-1:, :]], axis=0)
            down = K.concatenate([x[:1, :], x[:-1, :]], axis=0)
            left = K.concatenate([x[:, 1:], x[:, -1:]], axis=1)
            right = K.concatenate([x[:, :1], x[:, :-1]], axis=1)
        else:
            raise Exception("invalid shape")
        return up, down, left, right

    def call(self, inputs, alpha=1.2, epsilon=0.0001, *_):
        """
        :param inputs: (a, b), t = a * s + b -> g
        :param alpha:
        :param epsilon:
        :param _:
        """
        [a, b] = inputs
        l = self.luminance
        l_shifts = self.shift(l)
        a_shifts = self.shift(a)
        b_shifts = self.shift(b)
        loss = 0
        for i in range(4):  # 4 directions
            l_shift = l_shifts[i]
            a_shift = a_shifts[i]
            b_shift = b_shifts[i]
            smoothness = 1 / (K.pow(K.abs(l - l_shift), alpha) + epsilon)
            a_diff = K.sum(K.square(a - a_shift), axis=-1)
            b_diff = K.sum(K.square(b - b_shift), axis=-1)
            loss += smoothness * (a_diff + b_diff)
        loss = K.mean(K.batch_flatten(loss), axis=1, keepdims=True)
        return loss

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], 1)
        return output_shape


def gather_nd(arr, indices):
    """
    like tf.gather_nd
    https://www.tensorflow.org/api_docs/python/tf/gather_nd
    :param arr: ndarray
    :param indices: list of indices
    :return res: result

    example:
        arr = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ]

        indices1 = [2, 1]
        res1 = [
            [[9, 10], [11, 12]],
            [[5, 6], [7, 8]]
        ]

        indices2 = [
            [0, 0],
            [1, 1]
        ]
        res2 = [
            [1, 2],
            [7, 8]
        ]

        indices3 = [
            [0, 1, 0],
            [2, 0, 1]
        ]
        res3 = [3, 10]
    """
    n, dims = indices.shape
    shape = (n,) + arr.shape[dims:]
    res = np.zeros(shape, dtype=arr.dtype)
    for outer_idx in range(n):
        inner_res = arr
        for inner_idx in indices[outer_idx]:
            inner_res = inner_res[inner_idx]
        res[outer_idx] = inner_res
    return res


def calc_nearest_neighbors(s, fs, n_clusters=10, n_neighbors=8):
    """
    Cluster on source and find k nearest neighbor for each pixel
    :param s: (height, width, 3) ndarray, resized and normalized source image in range [0, 1]
    :param fs: (height, width, feature_channels) ndarray, feature map of source from VGG19
    :param n_clusters: number of clusters
    :param n_neighbors: number of neighbors
    :return p: (height * width * k, 2) ndarray, current pixel coordinates
    :return q: (height * width * k, 2) ndarray, neighbor pixel coordinates
    :return similarity: (height * width,) ndarray, similarity between pixel p and q
    """
    height, width, channels = fs.shape
    km = KMeans(n_clusters=n_clusters).fit(fs.reshape((-1, channels)))
    coordinate = np.zeros((height, width, 2), dtype=int)  # coordinate of each pixel
    for y in range(height):
        for x in range(width):
            coordinate[y, x] = y, x
    label = km.labels_.reshape((height, width))  # label of each pixel
    p_stack = []
    q_stack = []
    similarity_stack = []
    for i in range(n_clusters):
        # value for each pixel, [[r, g, b], ...]
        cluster_pixels = s[label == i]
        # coordinates for each pixel, [[y, x], ...]
        cluster_coords = coordinate[label == i]
        k = cluster_pixels.shape[0] - 1
        if k > n_neighbors:
            k = n_neighbors
        # nearest neighbors
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(cluster_pixels)
        # neighbor indices for each pixel, [[i0, i1, ..., ik], ...], i0 is index of the pixel
        indices = nn.kneighbors(cluster_pixels, return_distance=False)
        # get indices of neighbors for each pixel, then flatten it
        neighbor_indices = indices[:, 1:].reshape(-1)  # [i, ...]
        # flatten neighbor pixel coordinates
        q = cluster_coords[neighbor_indices]
        # repeat current pixel coords by k times to make it is corresponding with neighbor coords
        p = np.repeat(cluster_coords, k, axis=0)
        # sum of squared distance between pixel and its neighbor
        ssd = np.sum(np.square(gather_nd(s, p), gather_nd(s, q)), axis=-1)
        similarity = np.exp(1 - ssd) / k
        p_stack.append(p)
        q_stack.append(q)
        similarity_stack.append(similarity)
    p = np.concatenate(p_stack, axis=0)
    q = np.concatenate(q_stack, axis=0)
    similarity = np.concatenate(similarity_stack, axis=0)
    return p, q, similarity


class LossNL(Layer):
    """
    loss for pixel-wise matching difference
    """
    def __init__(self, p, q, similarity, **kwargs):
        """
        :param p: (height * width * k, 2) ndarray, current pixel coordinates
        :param q: (height * width * k, 2) ndarray, neighbor pixel coordinates
        :param similarity: (height * width,) ndarray, similarity between pixel p and q
        :param kwargs:
        """
        self.p = K.constant(p, dtype=tf.int32)
        self.q = K.constant(q, dtype=tf.int32)
        self.similarity = K.constant(similarity)
        super(LossNL, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LossNL, self).build(input_shape)

    def call(self, inputs, *_):
        """
        :param inputs: tt, transferred image
        """
        tt = inputs
        p = self.p
        q = self.q
        similarity = self.similarity

        def calc_loss(t):
            tp = tf.gather_nd(t, p)
            tq = tf.gather_nd(t, q)
            diff = K.sum(K.square(tp - tq), axis=-1)
            return similarity * diff
        loss = K.map_fn(calc_loss, tt)
        loss = K.mean(loss, axis=1)
        return loss

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], 1)
        return output_shape


def transfer_estimate(s, g, fs, fg, lambda_d=1., lambda_l=0.125, lambda_nl=2.0, lr=0.1, epochs=250):
    """
    :param s: (height, width, 3) ndarray, resized and normalized source image in range [0, 1]
    :param g: (height, width, 3) ndarray, resized and normalized guide image in range [0, 1]
    :param fs: (height, width, feature_channels) ndarray, feature map of source from VGG19
    :param fg: (height, width, feature_channels) ndarray, feature map of guide from VGG19
    :param lambda_d: weight for d loss
    :param lambda_l: weight for local constraint loss
    :param lambda_nl: weight for non-local constraint loss
    :param lr: learning rate
    :param epochs: training epochs
    :return t: transferred source
    :return a, b: t = a * s + b
    """
    # init
    a, b = None, None
    confidence = calc_confidence(fs, fg)
    luminance = calc_luminance(s)
    p, q, similarity = calc_nearest_neighbors(s, fs)
    # build model
    height, width, channels = fs.shape
    input_s = Input((height, width, 3), name="s")
    input_g = Input((height, width, 3), name="g")
    t, a, b = TransferLayer(a, b, name="color_transfer")(input_s)
    output_loss_d = LossD(confidence, name="loss_d")([t, input_g])
    output_loss_l = LossL(luminance, name="loss_l")([a, b])
    output_loss_nl = LossNL(p, q, similarity, name="loss_nl")(t)
    model = Model([input_s, input_g], [output_loss_d, output_loss_l, output_loss_nl])
    model.compile(optimizer=Adam(lr), loss=["MAE", "MAE", "MAE"], loss_weights=[lambda_d, lambda_l, lambda_nl])
    # train
    x = [np.expand_dims(s, axis=0), np.expand_dims(g, axis=0)]
    y = [np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))]
    model.fit(x, y, epochs=epochs)
    # color_transfer
    a, b = model.get_layer("color_transfer").get_weights()[2:]
    t = a * s + b
    return t, a, b
