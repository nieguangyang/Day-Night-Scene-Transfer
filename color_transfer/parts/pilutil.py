import numpy as np
from PIL import Image


def imread(filename):
    """
    :param filename: path to image
    :return img: (height, width, 3) ndarray of RGB image
    """
    im = Image.open(filename)
    if im.mode not in ("RGB", "L"):
        raise Exception("mode should be RGB or L")
    img = np.array(im, dtype=np.uint8)
    return img


def imresize(img, size, interp=2):
    """
    :param img: (height, width, 3) ndarray of RGB image
    :param size: (target_height, target_width)
    :param interp: 0 - nearest, 1 - lanczos, 2 - bilinear, 3 - cubic
    :return:
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    im = Image.fromarray(img)
    h, w = size
    im = im.resize((w, h), resample=interp)
    img = np.array(im)
    return img


def imsave(img, filename):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save(filename)


def preprocess(rgb, size, interp=2):
    """
    transform image into input to vgg19
    :param rgb: (height, width, 3) ndarray of RGB image
    :param size: (height, width)
    :param interp: 0 - nearest, 1 - lanczos, 2 - bilinear, 3 - cubic
    :return x: (1, 224, 224, 3) ndarray for input to vgg19
    """
    # resize
    rgb = imresize(rgb, size, interp)
    # RGB -> BGR
    bgr = rgb[:, :, ::-1].astype(np.float64)
    mean = (103.939, 116.779, 123.68)
    for i in range(3):
        bgr[:, :, i] -= mean[i]
    x = np.expand_dims(bgr, axis=0)
    return x
