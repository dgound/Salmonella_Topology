# Torus knot & link downsampling — cluster run guide

Self-contained pipeline: generate (p, q)-torus knots/links from the algebraic
construction on the Clifford torus, De Witt / uniform downsample, and analyze
topology preservation.

## Requirements

    numpy, scipy, pandas, matplotlib, seaborn, tqdm
    knoodle           # only needed for the KNOT analysis (Alexander invariants)

Links use a pure-numpy Gauss linking integral (no knoodle).

## Knots

Generate 1000 polygons (10k vertices) per knot, then analyze + plot.

    python generate_torus_knot_polygons.py --p 2 --q 3 --num-files 1000 -o knots/T_2_3
    python torus_knot_downsample_analyze.py -i knots/T_2_3 --p 2 --q 3 \
        -n 100 --no-replacement --method dewitt \
        --lengths 5,6,7,8,9,10,12,14,16,18,20,26,35,52,80,125,200,1000,2000,5000,10000 \
        -o knots/T_2_3/T23.csv
    python torus_knot_viz.py --csv "2,3:knots/T_2_3/T23.csv" -o knots/viz

- `--method dewitt` (one random point per chunk) or `uniform` (L random
  points from the whole chain).
- "correct type" = full Alexander fingerprint |Delta| at {-1, i, omega}.
- coprime (p, q) only.

## Links

Generate 1000 links (each component a 10k-vertex curve), analyze + plot.

    python generate_torus_link_polygons.py --p 2 --q 4 --num-files 1000 -o links/T_2_4
    python torus_link_downsample_analyze.py -i links/T_2_4 --p 2 --q 4 \
        -n 100 --no-replacement --method dewitt \
        --lengths 5,6,7,8,9,10,12,14,16,18,20,26,35,52,80,125,200,1000,2000,5000,10000 \
        -o links/T_2_4/T24.csv
    python torus_link_viz_suite.py --csv "2,4:links/T_2_4/T24.csv" -o links/viz

- gcd(p, q) = d gives a d-component link; pairwise |lk| = pq / d^2.
- The Gauss linking integral is O(L^2) in memory: an L=10000 pair builds a
  ~10000x10000 array (~2.4 GB), and each worker holds its own. On a
  many-core node cap the workers (e.g. OMP/pool size) so
  workers x (L^2 x 8 bytes x ~4) fits RAM at L=10000 and L=5000, or run the
  high-L lengths as a separate low-worker pass. `--no-replacement` keeps the
  sample count small at high L (floor(N/L): 1 at L=10000, 2 at 5000, 5 at
  2000, 10 at 1000), which bounds the wall-clock.

## Reproduce the exact experiment

Every polygon: **10 000 vertices** per component, **1000 instances** per
(p, q), random phase + random 3D rotation per instance (base seed 42,
per-instance seed = 42 + index). Downsampling: **100 subsamples per
length**, **--no-replacement**, and the dense length set

    L=5,6,7,8,9,10,11,12,14,16,18,20,23,26,30,35,40,46,52,60,70,80,90,100,125,150,200,1000,2000,5000,10000

Set `METHOD=dewitt` for the primary figures or `METHOD=uniform` for the
comparison figures.

### Knots — 10 coprime (p, q)

    (2,3) (2,5) (2,7) (2,9) (2,11) (3,4) (3,5) (4,5) (3,7) (5,6)

    L=5,6,7,8,9,10,11,12,14,16,18,20,23,26,30,35,40,46,52,60,70,80,90,100,125,150,200,1000,2000,5000,10000
    METHOD=dewitt
    ARGS=""
    for pq in "2 3" "2 5" "2 7" "2 9" "2 11" "3 4" "3 5" "4 5" "3 7" "5 6"; do
        p=${pq% *}; q=${pq#* }; D=knots/T_${p}_${q}
        python generate_torus_knot_polygons.py --p $p --q $q --num-files 1000 -o $D
        python torus_knot_downsample_analyze.py -i $D --p $p --q $q \
            -n 100 --no-replacement --method $METHOD --lengths $L -o $D/T${p}${q}.csv
        ARGS="$ARGS --csv ${p},${q}:$D/T${p}${q}.csv"
    done
    python torus_knot_viz.py $ARGS -o knots/viz

### Links — 10 (p, q) with gcd(p, q) > 1

    (2,2) (2,4) (2,6) (2,8) (2,10) (4,6) (3,3) (4,4) (5,5) (6,6)
    # components d = gcd(p,q):  2  2   2    2    2     2   3   4   5   6

    L=5,6,7,8,9,10,11,12,14,16,18,20,23,26,30,35,40,46,52,60,70,80,90,100,125,150,200,1000,2000,5000,10000
    METHOD=dewitt
    ARGS=""
    for pq in "2 2" "2 4" "2 6" "2 8" "2 10" "4 6" "3 3" "4 4" "5 5" "6 6"; do
        p=${pq% *}; q=${pq#* }; D=links/T_${p}_${q}
        python generate_torus_link_polygons.py --p $p --q $q --num-files 1000 -o $D
        python torus_link_downsample_analyze.py -i $D --p $p --q $q \
            -n 100 --no-replacement --method $METHOD --lengths $L -o $D/T${p}${q}.csv
        ARGS="$ARGS --csv ${p},${q}:$D/T${p}${q}.csv"
    done
    python torus_link_viz_suite.py $ARGS -o links/viz

Totals: 10 knot types x 1000 = 10 000 knot polygons; 10 link types x 1000
= 10 000 links (20 000-60 000 component polygons).

## Notes

- `--no-replacement` caps samples per length at floor(N/L) so no chain point
  is reused across the samples at a given length.
- All figures write both .png and .pdf.

