import numpy as np


class PatchMatch(object):
    """
    rewrite from https://github.com/rassilon712/Neural_Color_Transfer/blob/master/color_transfer.py
    """

    def __init__(self, a, b, patch_size=3):
        """
        patch match algorithm
        :param a: (ah, aw, channels) ndarray, image or feature map of A
        :param b: (bh, bw, channels) ndarray, image or feature map of B
        :param patch_size: patch size, normally 3 or 5
        """
        self.a = a.astype(np.float)
        self.b = b.astype(np.float)
        self.ah, self.aw = self.a.shape[:2]
        self.bh, self.bw = self.b.shape[:2]
        self.patch_size = patch_size
        self.nnf = np.zeros((self.ah, self.aw, 2), dtype=np.int)  # the nearest neighbour field, mapping from a to b
        self.nnd = np.zeros((self.ah, self.aw))  # the distance map for the nnf_computation
        self.initialise_nnf()

    def calc_dist(self, ay, ax, by, bx):
        """
        Calculate calc_dist between a patch in A to a patch in B.
        :return: Distance calculated between the two patches
        """
        d = self.patch_size // 2  # half patch size
        dy0 = min(ay, by, d)
        dy1 = min(self.ah - ay, self.bh - by, d + 1)
        dx0 = min(ax, bx, d)
        dx1 = min(self.aw - ax, self.bw - bx, d + 1)
        patch_a = self.a[ay - dy0:ay + dy1, ax - dx0:ax + dx1]
        patch_b = self.b[by - dy0:by + dy1, bx - dx0:bx + dx1]
        dist = np.sum(np.square(patch_a - patch_b)) / ((dy0 + dy1) * (dx0 + dx1))
        return dist

    def initialise_nnf(self):
        """
        Set up a random NNF
        Then calculate the calc_dists to fill up the NND
        """
        for ay in range(self.ah):
            for ax in range(self.aw):
                by = np.random.randint(self.bh)
                bx = np.random.randint(self.bw)
                self.nnf[ay, ax] = by, bx
                self.nnd[ay, ax] = self.calc_dist(ay, ax, by, bx)

    def improve_guess(self, ay, ax, by, bx, by_best, bx_best, dist_best):
        dist = self.calc_dist(ay, ax, by, bx)
        if dist < dist_best:
            by_best, bx_best, dist_best = by, bx, dist
        return by_best, bx_best, dist_best

    def improve_nnf(self, total_iter=5):
        """
        Optimize the NNF using PatchMatch Algorithm
        :param total_iter: number of iterations
        """
        for it in range(1, total_iter + 1):
            if it % 2 == 0:
                ay_start, ay_end = 0, self.ah
                ax_start, ax_end = 0, self.aw
                step = 1
            else:
                ay_start, ay_end = self.ah - 1, -1
                ax_start, ax_end = self.aw - 1, -1
                step = -1
            for ay in range(ay_start, ay_end, step):
                for ax in range(ax_start, ax_end, step):
                    by_best, bx_best = self.nnf[ay, ax]
                    dist_best = self.nnd[ay, ax]
                    # propagate
                    if 0 <= ay - step < self.ah:
                        by, bx = self.nnf[ay - step, ax]  # neighbor
                        by += step
                        if 0 <= by < self.bh:
                            by_best, bx_best, dist_best = self.improve_guess(ay, ax, by, bx, by_best, bx_best, dist_best)
                    if 0 <= ax - step < self.aw:
                        by, bx = self.nnf[ay, ax - step]  # neighbor
                        bx += step
                        if 0 <= bx < self.bw:
                            by_best, bx_best, dist_best = self.improve_guess(ay, ax, by, bx, by_best, bx_best, dist_best)
                    # random search
                    r = max(self.ah, self.aw)  # radius
                    while r >= 1:
                        by_min = max(by_best - r, 0)
                        by_max = min(by_best + r + 1, self.bh)
                        bx_min = max(bx_best - r, 0)
                        bx_max = min(bx_best + r + 1, self.bw)
                        by = np.random.randint(by_min, by_max)
                        bx = np.random.randint(bx_min, bx_max)
                        by_best, bx_best, dist_best = self.improve_guess(ay, ax, by, bx, by_best, bx_best, dist_best)
                        r = r // 2
                    self.nnf[ay, ax] = by_best, bx_best
                    self.nnd[ay, ax] = dist_best
            print("iteration: %d/%d" % (it, total_iter))


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
    sr = PatchMatch(src, ref, patch_size)
    sr.improve_nnf(total_iter)
    nnf_sr = sr.nnf
    rs = PatchMatch(ref, src, patch_size)
    rs.improve_nnf(total_iter)
    nnf_rs = rs.nnf
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
