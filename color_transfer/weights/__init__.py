import os

PATH = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
# weights of VGG19 pre-trained on ImageNet
# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
VGG19_IMAGENET_WEIGHTS = PATH + "/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
# weights of PSPNet50 pre-trained on ADE20K
# https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5
PSPNET50_ADE20K_WEIGHTS = PATH + "/pspnet50_ade20k.h5"
