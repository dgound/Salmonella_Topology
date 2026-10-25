// C entry point for loading the genomic Moran's I calculator via ctypes.
#include "morans_i_genomic.h"
#include <exception>
#include <limits>
#include <vector>

extern "C" int morans_i_cloud(const double* xyz, const int* tp, int n,
                              double cycle_bins, unsigned long long seed,
                              double* out) {
    try {
        std::vector<Point3D> points;
        points.reserve(n);
        for (int i = 0; i < n; ++i) {
            points.emplace_back(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
        }
        std::vector<int> timepoints(tp, tp + n);
        MoransIResult result = GenomicMoransICalculator(cycle_bins)
            .compute_genomic_morans_i_with_pvalue(points, timepoints, seed);
        out[0] = result.morans_i;
        out[1] = result.p_value;
        return 0;
    } catch (const std::exception&) {
        out[0] = std::numeric_limits<double>::quiet_NaN();
        out[1] = std::numeric_limits<double>::quiet_NaN();
        return 1;
    }
}
