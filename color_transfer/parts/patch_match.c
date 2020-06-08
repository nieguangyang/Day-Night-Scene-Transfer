#include <stdlib.h>
#include <stdio.h>

int min(int x, int y)
{
    if (x <= y)
    {
        return x;
    }
    return y;
}

int max(int x, int y)
{
    if (x >= y)
    {
        return x;
    }
    return y;
}

double calc_dist
(
    double *a, int ah, int aw,
    double *b, int bh, int bw,
    int channels, int patch_size,
    int ay, int ax, int by, int bx
)
{
    int y, x, ch;
    int d = patch_size / 2;
    double dist = 0.;
    int n_pixels = 0;
    double temp_dist;  // square distance between two pixel at a channel
    int _ay, _ax, _by, _bx;
    double av, bv;
    int i;
    for(y = -d; y <= d; y++)
    {
        for(x = -d; x <= d; x++)
        {
            _ay = ay + y;
            _ax = ax + x;
            _by = by + y;
            _bx = bx + x;
            if (_ay < 0 || _ay >= ah || _ax < 0 || _ax >= aw || _by < 0 || _by >= bh || _bx < 0 || _bx >= bw)
            {
                continue;
            }
            for(ch = 0; ch < channels; ch++)
            {
                av = a[(_ay * aw + _ax) * channels + ch];
                bv = b[(_by * bw + _bx) * channels + ch];
                temp_dist = av - bv;
                dist += temp_dist * temp_dist;
            }
            n_pixels++;
        }
    }
    dist /= (double)n_pixels;
    return dist;
}

void improve_guess
(
    double *a, int ah, int aw,
    double *b, int bh, int bw,
    int channels, int patch_size,
    int ay, int ax, int by, int bx,
    int *by_best, int *bx_best, double *dist_best
)
{
    double dist = calc_dist(a, ah, aw, b, bh, bw, channels, patch_size, ay, ax, by, bx);
    if(dist < *dist_best)
    {
        *by_best = by;
        *bx_best = bx;
        *dist_best = dist;
    }
}

int randint(int low, int high)  // same as randint in python
{
    int n = rand() % (high - low + 1) + low;
    return n;
}

// a, b, nnf, nnd are all flattened
void patch_match
(
    double *a, int ah, int aw,
    double *b, int bh, int bw,
    int channels, int patch_size,
    int *nnf, double *nnd,
    int total_iter
)
{
    // init
    int ay, ax, by, bx;
    for(ay = 0; ay < ah; ay++)
    {
        for(ax = 0; ax < aw; ax++)
        {
            by = randint(0, bh - 1);
            bx = randint(0, bw - 1);
            nnf[(ay * aw + ax) * 2 + 0] = by;
            nnf[(ay * aw + ax) * 2 + 1] = bx;
            nnd[ay * aw + ax] = calc_dist(a, ah, aw, b, bh, bw, channels, patch_size, ay, ax, by, bx);
        }
    }
    // propagate and random search
    int step;
    int by_best, bx_best;
    double dist_best;
    int r;
    int by_min, by_max, bx_min, bx_max;
    int iter;
    for(iter = 0; iter < total_iter; iter++)
    {
        if(iter % 2 == 0)
        {
            ay = 0;
            step = 1;
        }
        else
        {
            ay = ah - 1;
            step = -1;
        }
        while(ay >= 0 && ay < ah)
        {
            if(iter % 2 == 0)
            {
                ax = 0;
            }
            else
            {
                ax = aw - 1;
            }
            while(ax >= 0 && ax < aw)
            {
                by_best = nnf[(ay * aw + ax) * 2 + 0];
                bx_best = nnf[(ay * aw + ax) * 2 + 1];
                dist_best = nnd[ay * aw + ax];
                // propagate
                if(ay - step >= 0 && ay - step < ah)
                {
                    by = nnf[((ay - step) * aw + ax) * 2 + 0] + step;
                    bx = nnf[((ay - step) * aw + ax) * 2 + 1];
                    if(by >= 0 && by < bh)
                    {
                        improve_guess(a, ah, aw, b, bh, bw, channels, patch_size, ay, ax, by, bx, &by_best, &bx_best, &dist_best);
                    }
                }
                if(ax - step >= 0 && ax - step < aw)
                {
                    by = nnf[(ay * aw + (ax - step)) * 2 + 0];
                    bx = nnf[(ay * aw + (ax - step)) * 2 + 1] + step;
                    if(bx >= 0 && bx < bw)
                    {
                        improve_guess(a, ah, aw, b, bh, bw, channels, patch_size, ay, ax, by, bx, &by_best, &bx_best, &dist_best);
                    }
                }
                // random search
                r = max(ah, aw);
                while(r >= 1)
                {
                    by_min = max(by_best - r, 0);
                    by_max = min(by_best + r + 1, bh);
                    bx_min = max(bx_best - r, 0);
                    bx_max = min(bx_best + r + 1, bw);
                    by = randint(by_min, by_max);
                    bx = randint(bx_min, bx_max);
                    improve_guess(a, ah, aw, b, bh, bw, channels, patch_size, ay, ax, by, bx, &by_best, &bx_best, &dist_best);
                    r /= 2;
                }
                nnf[(ay * aw + ax) * 2 + 0] = by_best;
                nnf[(ay * aw + ax) * 2 + 1] = bx_best;
                nnd[ay * aw + ax] = dist_best;
                ax += step;
            }
            ay += step;
        }
        printf("iteration: %d/%d\n", iter, total_iter);
    }
}

