/*
 * morans_i_c.c -- fast Moran's I permutation test for single-curve walks.
 *
 * Compile:
 *   cc -O2 -shared -fPIC -lm -o morans_i_c.so morans_i_c.c
 *
 * Algorithm matches morans_i.py exactly (ported from the C++ reference):
 *   1. Inverse-squared-distance spatial weights, optionally excluding
 *      circular chain neighbors.
 *   2. Row-standardize.
 *   3. Circular deviation of genomic positions, standardized to unit variance.
 *   4. I = (N / W_total) * (d^T W d) / (d^T d).
 *   5. Permutation test: shuffle genomic positions M times, count I_perm >= I_obs.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define EPS 1e-10

/* ------------------------------------------------------------------ */
/* splitmix64 RNG (one uint64 state, good quality for shuffles)       */
/* ------------------------------------------------------------------ */
typedef struct { uint64_t s; } rng_t;

static inline uint64_t rng_next(rng_t *r)
{
    uint64_t z = (r->s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Fisher-Yates shuffle in-place */
static void shuffle_int(int *a, int n, rng_t *r)
{
    for (int i = n - 1; i > 0; i--) {
        uint64_t j = rng_next(r) % (uint64_t)(i + 1);
        int tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
}

/* ------------------------------------------------------------------ */
/* Weight matrix: inverse squared distance, optional chain exclusion  */
/* ------------------------------------------------------------------ */
static void compute_weights(const double *coords, int N, int exclude_leq,
                            double *W, double *W_total_out)
{
    for (int i = 0; i < N; i++) {
        W[i * N + i] = 0.0;
        const double xi = coords[i * 3], yi = coords[i * 3 + 1],
                     zi = coords[i * 3 + 2];
        for (int j = i + 1; j < N; j++) {
            double dx = xi - coords[j * 3];
            double dy = yi - coords[j * 3 + 1];
            double dz = zi - coords[j * 3 + 2];
            double w = 1.0 / (dx * dx + dy * dy + dz * dz + EPS);
            W[i * N + j] = w;
            W[j * N + i] = w;
        }
    }

    if (exclude_leq > 0) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int d = abs(i - j);
                if ((d < N - d ? d : N - d) <= exclude_leq)
                    W[i * N + j] = 0.0;
            }
    }

    double wtot = 0.0;
    for (int i = 0; i < N; i++) {
        double rs = 0.0;
        for (int j = 0; j < N; j++) rs += W[i * N + j];
        if (rs > EPS) {
            double inv = 1.0 / rs;
            for (int j = 0; j < N; j++) W[i * N + j] *= inv;
        }
        wtot += rs > EPS ? 1.0 : 0.0;   /* post-standardization row sum is 1 */
    }
    *W_total_out = wtot;
}

/* ------------------------------------------------------------------ */
/* Circular standardization                                           */
/* ------------------------------------------------------------------ */
/* Returns 0 on success, -1 if degenerate (zero variance). */
static int circular_standardize(const int *gpos, int N, double two_pi_cb,
                                double *d)
{
    double sc = 0.0, ss = 0.0;
    for (int i = 0; i < N; i++) {
        double a = gpos[i] * two_pi_cb;
        sc += cos(a);
        ss += sin(a);
    }
    double inv_n = 1.0 / (double)N;
    double mc = sc * inv_n, ms = ss * inv_n;
    double R = sqrt(mc * mc + ms * ms);
    double xbar = R < 1e-12 ? 0.0 : atan2(ms, mc);
    double var = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = gpos[i] * two_pi_cb - xbar;
        double di = atan2(sin(diff), cos(diff));
        d[i] = di;
        var += di * di;
    }
    double sd = sqrt(var * inv_n);
    if (sd <= EPS) return -1;
    double inv_sd = 1.0 / sd;
    for (int i = 0; i < N; i++) d[i] *= inv_sd;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Core formula: I = (N / W_total) * (d^T W d) / (d^T d)             */
/* ------------------------------------------------------------------ */
static double morans_i_core(const double *W, const double *d, int N,
                            double W_total)
{
    double numer = 0.0, denom = 0.0;
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int j = 0; j < N; j++) s += W[i * N + j] * d[j];
        numer += d[i] * s;
        denom += d[i] * d[i];
    }
    return ((double)N / W_total) * (numer / denom);
}

/* ------------------------------------------------------------------ */
/* Public API: single-walk permutation test                           */
/*                                                                    */
/* out[0] = I_observed                                                */
/* out[1] = P_value                                                   */
/* out[2] = empirical_mean                                            */
/* out[3] = empirical_std                                             */
/* out[4] = n_permutations_valid                                      */
/* ------------------------------------------------------------------ */
void morans_i_permtest(const double *coords, const int *gpos,
                       int N, int cycle_bins, int n_perms,
                       int exclude_leq, uint64_t seed, double *out)
{
    double *W    = (double *)malloc((size_t)N * N * sizeof(double));
    double *d    = (double *)malloc((size_t)N * sizeof(double));
    int    *perm = (int *)malloc((size_t)N * sizeof(int));

    double W_total;
    compute_weights(coords, N, exclude_leq, W, &W_total);

    if (W_total <= EPS || circular_standardize(gpos, N,
            2.0 * M_PI / (double)cycle_bins, d) != 0) {
        out[0] = out[1] = out[2] = out[3] = NAN;
        out[4] = 0.0;
        free(W); free(d); free(perm);
        return;
    }

    double I_obs = morans_i_core(W, d, N, W_total);
    double two_pi_cb = 2.0 * M_PI / (double)cycle_bins;

    rng_t rng = { seed };
    int count_ge = 0, n_valid = 0;
    double sp = 0.0, sp2 = 0.0;

    for (int p = 0; p < n_perms; p++) {
        memcpy(perm, gpos, (size_t)N * sizeof(int));
        shuffle_int(perm, N, &rng);

        if (circular_standardize(perm, N, two_pi_cb, d) != 0)
            continue;

        double Ip = morans_i_core(W, d, N, W_total);
        n_valid++;
        sp  += Ip;
        sp2 += Ip * Ip;
        if (Ip >= I_obs) count_ge++;
    }

    out[0] = I_obs;
    out[1] = (double)(count_ge + 1) / (double)(n_perms + 1);
    if (n_valid > 0) {
        double mean = sp / n_valid;
        out[2] = mean;
        out[3] = sqrt(sp2 / n_valid - mean * mean);
    } else {
        out[2] = out[3] = NAN;
    }
    out[4] = (double)n_valid;

    free(W); free(d); free(perm);
}

/* ------------------------------------------------------------------ */
/* Public API: batch over n_walks walks                               */
/*                                                                    */
/* all_coords : n_walks * N * 3, row-major (all walks same N)         */
/* gpos       : N ints (shared across walks)                          */
/* out        : n_walks * 5                                           */
/* ------------------------------------------------------------------ */
void morans_i_batch(const double *all_coords, const int *gpos,
                    int N, int n_walks, int cycle_bins, int n_perms,
                    int exclude_leq, uint64_t seed, double *out)
{
    for (int w = 0; w < n_walks; w++) {
        uint64_t wseed = seed ^ ((uint64_t)w * 0x517cc1b727220a95ULL);
        morans_i_permtest(all_coords + (size_t)w * N * 3,
                          gpos, N, cycle_bins, n_perms, exclude_leq,
                          wseed, out + (size_t)w * 5);
    }
}
