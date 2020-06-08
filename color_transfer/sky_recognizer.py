import numpy as np

from color_transfer.parts.pspnet50_ade20k import pspnet50_ade20k, preprocess, resize


class SkyRecognizer:
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

    def recognize(self, img):
        """
        :param img: (height, width, channels) ndarray
        :return mask: (height, width) ndarray, 1 for sky, 0 for other
        """
        label = self.predict(img)
        index = np.argmax(label, axis=2)
        mask = np.uint8(index == 2)
        return mask


def test():
    import os
    import matplotlib.pyplot as plt

    from color_transfer.parts.pilutil import imread

    path = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1]) + "/parts/test/pspnet50_ade20k"
    pattern = path + "/%d.jpg"
    weights = "E:/ai/weights/pretrained/pspnet50_ade20k.h5"
    sr = SkyRecognizer(weights)
    for i in range(10):
        file = pattern % i
        img = imread(file)
        mask = sr.recognize(img)
        ax = plt.subplot(121)
        ax.imshow(img)
        ax = plt.subplot(122)
        ax.imshow(mask)
        plt.show()


if __name__ == "__main__":
    test()
