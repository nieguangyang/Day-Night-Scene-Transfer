import os
import numpy as np
from platform import system, architecture
from ctypes import CDLL, c_int, c_double


path = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
# gcc -fPIC -shared -m64 patch_match.c -o patch_match_win_64.so
patch_match_win_64 = path + "/patch_match_win_64.so"
# gcc -fPIC -shared -m64 patch_match.c -o patch_match_linux_64.so
patch_match_linux_64 = path + "/patch_match_linux_64.so"
# gcc -fPIC -shared -m64 patch_match.c -o patch_match_mac_64.so
patch_match_mac_64 = path + "/patch_match_mac_64.so"


# void patch_match
# (
#     float *a, int ah, int aw,
#     float *b, int bh, int bw,
#     int channels, int patch_size,
#     int *nnf, float *nnd,
#     int total_iter
# )
def patch_match(a, b, patch_size=3, total_iter=5):
    sys = system()
    bits = architecture()[0]
    if bits == "64bit":
        if sys == "Windows":
            patch_match_lib = patch_match_win_64
        elif sys == "Linux":
            patch_match_lib = patch_match_linux_64
        else:
            # raise Exception("No support for Mac yet.")
            patch_match_lib = patch_match_mac_64
    else:
        raise Exception("No support for 32bit.")
    patch_match_c = CDLL(patch_match_lib)
    patch_match_c.calc_dist.restype = c_double
    patch_match_c.improve_guess.restype = None
    patch_match_c.patch_match.restype = None
    ah, aw, channels = a.shape
    bh, bw, _channels = b.shape
    assert channels == _channels
    a = a.astype(np.float).flatten().tolist()
    b = b.astype(np.float).flatten().tolist()
    c_a, c_ah, c_aw = (c_double * len(a))(*a), c_int(ah), c_int(aw)
    c_b, c_bh, c_bw = (c_double * len(b))(*b), c_int(bh), c_int(bw)
    c_channels, c_patch_size = c_int(channels), c_int(patch_size)
    size_nnf = ah * aw * 2
    size_nnd = ah * aw
    c_nnf = (c_int * size_nnf)()
    c_nnd = (c_double * size_nnd)()
    c_total_iter = c_int(total_iter)
    patch_match_c.patch_match(
        c_a, c_ah, c_aw,
        c_b, c_bh, c_bw,
        c_channels, c_patch_size,
        c_nnf, c_nnd,
        c_total_iter
    )
    nnf = np.array(c_nnf[:]).reshape((ah, aw, 2))
    return nnf


def nn_search(src, ref, patch_size=3, total_iter=5):
    """
    Search nearest neighbor bidirectionally
    :param src: (ah, aw, channels) ndarray, image or feature map of source
    :param ref: (bh, bw, channels) ndarray, image or feature map of reference
    :param patch_size: patch size
    :param total_iter: total iterations for patch match
    :return nnf_sr: src->ref forward NNF
    :return nnf_rs: ref->src backward NNF
    """
    nnf_sr = patch_match(src, ref, patch_size, total_iter)
    nnf_rs = patch_match(ref, src, patch_size, total_iter)
    return nnf_sr, nnf_rs


def bds_vote(ref, nnf_sr, nnf_rs, patch_size=3, w=8):
    """
    Reconstruct image or feature map by bidirectional similarity voting
    :param ref: image or feature map of reference
    :param nnf_sr: src->ref forward NNF (nearest neighbor field)
    :param nnf_rs: ref->src backward NNF
    :param patch_size: patch size, should be odd
    :param w: weight for completeness, i.e. backward NNF
    :return guide: reconstructed image or feature map
    """
    sh, sw = nnf_sr.shape[:2]
    rh, rw, channels = ref.shape
    guide = np.zeros((sh, sw, channels))
    weight = np.zeros((sh, sw, channels))
    ws = 1 / (sh * sw)  # weight for a
    wr = w / (rh * rw)  # weight for b
    d = patch_size // 2  # half patch size
    # coherence, a->b forward NNF enforces coherence
    for ay in range(sh):
        for ax in range(sw):
            by, bx = nnf_sr[ay, ax]
            dy0 = min(ay, by, d)
            dy1 = min(sh - ay, rh - by, d + 1)
            dx0 = min(ax, bx, d)
            dx1 = min(sw - ax, rw - bx, d + 1)
            guide[ay - dy0:ay + dy1, ax - dx0:ax + dx1, :] += ws * ref[by - dy0:by + dy1, bx - dx0:bx + dx1, :]
            weight[ay - dy0:ay + dy1, ax - dx0:ax + dx1, :] += ws
    # completeness, b->a backward NNF enforces completeness
    for by in range(rh):
        for bx in range(rw):
            ay, ax = nnf_rs[by, bx]
            dy0 = min(ay, by, d)
            dy1 = min(sh - ay, rh - by, d + 1)
            dx0 = min(ax, bx, d)
            dx1 = min(sw - ax, rw - bx, d + 1)
            guide[ay - dy0:ay + dy1, ax - dx0:ax + dx1, :] += wr * ref[by - dy0:by + dy1, bx - dx0:bx + dx1, :]
            weight[ay - dy0:ay + dy1, ax - dx0:ax + dx1, :] += wr
    weight[weight == 0] = 1
    guide /= weight
    return guide


def test():
    from color_transfer.parts.pilutil import imread
    from random import randint

    from color_transfer.parts.nnf_computation import PatchMatch

    img_src = "./test/nn_computation/a.jpg"
    img_ref = "./test/nn_computation/b.jpg"
    src = imread(img_src)
    ref = imread(img_ref)

    from color_transfer.parts.pilutil import imresize
    size = (224, 224)
    src = imresize(src, size)
    ref = imresize(ref, size)

    patch_size = 3
    a, b = src, ref
    pm = PatchMatch(a, b, patch_size=patch_size)
    patch_match_c = CDLL("E:/patch_match.so")
    calc_dist = patch_match_c.calc_dist
    calc_dist.restype = c_double
    ah, aw, channels = a.shape
    bh, bw, _channels = b.shape
    assert channels == _channels
    h, w = a.shape[:2]
    a = a.flatten().tolist()
    b = b.flatten().tolist()
    c_a, c_ah, c_aw = (c_double * len(a))(*a), c_int(ah), c_int(aw)
    c_b, c_bh, c_bw = (c_double * len(b))(*b), c_int(bh), c_int(bw)
    c_channels, c_patch_size = c_int(channels), c_int(patch_size)
    for _ in range(10):
        ay, ax = randint(0, h - 1), randint(0, w - 1)
        by, bx = randint(0, h - 1), randint(0, w - 1)
        dist_py = pm.calc_dist(ay, ax, by, bx)
        c_ay, c_ax, c_by, c_bx = c_int(ay), c_int(ax), c_int(by), c_int(bx)
        dist_c = calc_dist(
            c_a, c_ah, c_aw,
            c_b, c_bh, c_bw,
            c_channels, c_patch_size,
            c_ay, c_ax, c_by, c_bx
        )
        print(dist_py, dist_c)

    from ctypes import byref
    improve_guess = patch_match_c.improve_guess
    improve_guess.restype = None
    ay, ax, by, bx = 100, 100, 99, 99
    by_best, bx_best, dist_best = 100, 100, 1000000
    c_ay, c_ax, c_by, c_bx = c_int(ay), c_int(ax), c_int(by), c_int(bx)
    c_by_best, c_bx_best, c_dist_best = c_int(by_best), c_int(bx_best), c_double(dist_best)
    improve_guess(
        c_a, c_ah, c_aw,
        c_b, c_bh, c_bw,
        c_channels, c_patch_size,
        c_ay, c_ax, c_by, c_bx,
        byref(c_by_best), byref(c_bx_best), byref(c_dist_best)
    )
    print(c_by_best.value, c_bx_best.value, c_dist_best.value)

    total_iter = 0
    size_nnf = ah * aw * 2
    size_nnd = ah * aw
    c_nnf = (c_int * size_nnf)()
    c_nnd = (c_double * size_nnd)()
    c_total_iter = c_int(total_iter)
    patch_match_c.patch_match.restype = None
    patch_match_c.patch_match(
        c_a, c_ah, c_aw,
        c_b, c_bh, c_bw,
        c_channels, c_patch_size,
        c_nnf, c_nnd,
        c_total_iter
    )
    print(c_nnd[:])


if __name__ == "__main__":
    test()
