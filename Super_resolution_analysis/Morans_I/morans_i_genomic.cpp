#include "morans_i_genomic.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <future>

void GenomicMoransICalculator::validate_input(const std::vector<Point3D>& spatial_coords,
                                             const std::vector<int>& genomic_positions) const {
    if (spatial_coords.empty()) {
        throw std::invalid_argument("Spatial coordinates cannot be empty");
    }
    
    if (spatial_coords.size() != genomic_positions.size()) {
        throw std::invalid_argument("Spatial coordinates and genomic positions must have same size");
    }
    
    if (spatial_coords.size() < 3) {
        throw std::invalid_argument("Need at least 3 points for Moran's I calculation");
    }
}

// Circular deviations of timepoint angles from their circular mean; returns their RMS
static double circular_deviations(const std::vector<int>& positions, double cycle_bins,
                                  std::vector<double>& d_theta_vals) {
    const size_t n = positions.size();
    d_theta_vals.resize(n);
    
    double sum_cos = 0.0, sum_sin = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double angle = positions[i] * 2.0 * M_PI / cycle_bins;
        d_theta_vals[i] = angle;
        sum_cos += std::cos(angle);
        sum_sin += std::sin(angle);
    }
    double mean_cos = sum_cos / n;
    double mean_sin = sum_sin / n;
    double X_bar = std::atan2(mean_sin, mean_cos);
    
    double std_dev = 0.0;
    for (double& val : d_theta_vals) {
        double diff = val - X_bar;
        val = std::atan2(std::sin(diff), std::cos(diff));
        std_dev += val * val;
    }
    return std::sqrt(std_dev / n);
}

double GenomicMoransICalculator::compute_genomic_morans_i(
    const std::vector<Point3D>& spatial_coords,
    const std::vector<int>& genomic_positions) const {
    
    validate_input(spatial_coords, genomic_positions);
    
    const size_t n = spatial_coords.size();
    std::cout << "Computing genomic Moran's I for " << n << " points..." << std::endl;
    
    std::cout << "Computing circular genomic deviations..." << std::endl;
    std::vector<double> d_theta_vals;
    double std_dev = circular_deviations(genomic_positions, cycle_bins_, d_theta_vals);
    
    if (std_dev > EPSILON) {
        for (double& val : d_theta_vals) {
            val /= std_dev;
        }
    } else {
        throw std::runtime_error("Standard deviation of circular deviations is zero");
    }
    
    std::cout << "Computing final Moran's I value..." << std::endl;
    std::vector<double> numerator;
    double W = stream_quadratic_forms(spatial_coords, d_theta_vals, 1, numerator);
    
    if (W <= EPSILON) {
        throw std::runtime_error("Sum of weights is zero or too small");
    }
    
    double denominator = 0.0;
    for (double val : d_theta_vals) {
        denominator += val * val;
    }
    
    if (denominator <= EPSILON) {
        throw std::runtime_error("Denominator in Moran's I calculation is zero");
    }
    
    // I = (N / W) * (numerator / denominator)
    double morans_i = (static_cast<double>(n) / W) * (numerator[0] / denominator);
    return morans_i;
}

double GenomicMoransICalculator::stream_quadratic_forms(
    const std::vector<Point3D>& spatial_coords,
    const std::vector<double>& D,
    size_t ncol,
    std::vector<double>& num_out) const {
    
    const size_t n = spatial_coords.size();
    std::vector<double> xs(n), ys(n), zs(n);
    for (size_t i = 0; i < n; ++i) {
        xs[i] = spatial_coords[i].x;
        ys[i] = spatial_coords[i].y;
        zs[i] = spatial_coords[i].z;
    }
    
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "  Detected " << num_threads << " CPU threads available" << std::endl;
    const size_t threads_to_use = std::max<size_t>(1, std::min<size_t>(num_threads, n));
    std::cout << "  Streaming spatial weights over " << n << " rows x " << ncol
              << " columns using " << threads_to_use << " threads..." << std::endl;
    
    // Row standardization is applied per row after its sum r_i is known, so W is never stored
    auto worker = [&](size_t row_begin, size_t row_end, std::vector<double>& num) -> double {
        const double* __restrict x = xs.data();
        const double* __restrict y = ys.data();
        const double* __restrict z = zs.data();
        const double* __restrict Dp = D.data();
        std::vector<double> partial(ncol);
        double* __restrict p = partial.data();
        double* __restrict nm = num.data();
        double W = 0.0;
        
        for (size_t i = row_begin; i < row_end; ++i) {
            std::fill(partial.begin(), partial.end(), 0.0);
            const double xi = x[i], yi = y[i], zi = z[i];
            double r_i = 0.0;
            
            for (size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                const double dx = xi - x[j];
                const double dy = yi - y[j];
                const double dz = zi - z[j];
                const double w = 1.0 / (dx * dx + dy * dy + dz * dz + EPSILON);
                r_i += w;
                const double* __restrict Dj = Dp + j * ncol;
                for (size_t k = 0; k < ncol; ++k) {
                    p[k] += w * Dj[k];
                }
            }
            
            const double* __restrict Di = Dp + i * ncol;
            if (r_i > EPSILON) {
                for (size_t k = 0; k < ncol; ++k) {
                    nm[k] += Di[k] * p[k] / r_i;
                }
                W += 1.0;
            } else {
                for (size_t k = 0; k < ncol; ++k) {
                    nm[k] += Di[k] * p[k];
                }
                W += r_i;
            }
        }
        return W;
    };
    
    std::vector<std::vector<double>> thread_num(threads_to_use, std::vector<double>(ncol, 0.0));
    std::vector<std::future<double>> futures;
    const size_t rows_per_thread = n / threads_to_use;
    const size_t remaining_rows = n % threads_to_use;
    size_t row_begin = 0;
    
    for (size_t t = 0; t < threads_to_use; ++t) {
        size_t row_end = row_begin + rows_per_thread + (t < remaining_rows ? 1 : 0);
        futures.push_back(std::async(std::launch::async, worker,
            row_begin, row_end, std::ref(thread_num[t])));
        row_begin = row_end;
    }
    
    double W = 0.0;
    num_out.assign(ncol, 0.0);
    for (size_t t = 0; t < threads_to_use; ++t) {
        W += futures[t].get();
        for (size_t k = 0; k < ncol; ++k) {
            num_out[k] += thread_num[t][k];
        }
    }
    return W;
}

double GenomicMoransICalculator::calculate_morans_i_core(
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>& standardized_variable) const {
    
    const size_t n = weights.size();
    
    // Compute sum of weights
    double W = 0.0;
    for (const auto& row : weights) {
        W += std::accumulate(row.begin(), row.end(), 0.0);
    }
    
    if (W <= EPSILON) {
        throw std::runtime_error("Sum of weights is zero or too small");
    }
    
    // numerator = variable^T * weights * variable
    double numerator = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            numerator += standardized_variable[i] * weights[i][j] * standardized_variable[j];
        }
    }
    
    // denominator = sum(variable^2)
    double denominator = 0.0;
    for (double val : standardized_variable) {
        denominator += val * val;
    }
    
    if (denominator <= EPSILON) {
        throw std::runtime_error("Denominator in Moran's I calculation is zero");
    }
    
    // I = (N / W) * (numerator / denominator)
    double morans_i = (static_cast<double>(n) / W) * (numerator / denominator);
    return morans_i;
}

double GenomicMoransICalculator::calculate_morans_i_variance(
    const std::vector<std::vector<double>>& weights) const {
    
    const size_t n = weights.size();
    
    // Calculate S0 (sum of all weights)
    double S0 = 0.0;
    for (const auto& row : weights) {
        S0 += std::accumulate(row.begin(), row.end(), 0.0);
    }
    
    // Calculate S1 (sum of squared weights)
    double S1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double w_ij = weights[i][j];
            S1 += w_ij * w_ij;
        }
    }
    S1 *= 0.5; // Each weight counted twice in double loop
    
    // Calculate S2 (sum of squared row/column sums)
    double S2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double row_sum = std::accumulate(weights[i].begin(), weights[i].end(), 0.0);
        double col_sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            col_sum += weights[j][i];
        }
        S2 += (row_sum + col_sum) * (row_sum + col_sum);
    }
    
    // Calculate variance using standard formula
    double n_d = static_cast<double>(n);
    double term1 = (n_d * n_d - 3.0 * n_d + 3.0) * S1;
    double term2 = -1.0 * n_d * S2;
    double term3 = 3.0 * S0 * S0;
    
    double variance = (term1 + term2 + term3) / ((n_d - 1.0) * (n_d - 2.0) * (n_d - 3.0) * S0 * S0);
    
    // Subtract the squared expected value
    double expected_i = -1.0 / (n - 1);
    variance -= expected_i * expected_i;
    
    return variance;
}

MoransIResult GenomicMoransICalculator::compute_genomic_morans_i_with_pvalue(
    const std::vector<Point3D>& spatial_coords,
    const std::vector<int>& genomic_positions,
    unsigned long long seed) const {

    validate_input(spatial_coords, genomic_positions);

    std::cout << "Computing genomic Moran's I with p-value..." << std::endl;
    
    const size_t n = spatial_coords.size();
    const size_t ncol = 1 + PERMUTATIONS;
    std::cout << "Computing genomic Moran's I for " << n << " points..." << std::endl;
    
    // Column 0: observed standardized deviations, columns 1..PERMUTATIONS: shuffled timepoints
    std::cout << "Computing circular genomic deviations..." << std::endl;
    std::vector<double> D(n * ncol);
    std::vector<double> d_theta_vals;
    double std_dev = circular_deviations(genomic_positions, cycle_bins_, d_theta_vals);
    
    if (std_dev <= EPSILON) {
        throw std::runtime_error("Standard deviation of circular deviations is zero");
    }
    for (size_t i = 0; i < n; ++i) {
        D[i * ncol] = d_theta_vals[i] / std_dev;
    }
    
    std::random_device rd;
    std::mt19937 gen(seed != 0 ? static_cast<std::mt19937::result_type>(seed) : rd());
    std::vector<int> permuted_positions = genomic_positions;
    
    for (size_t k = 1; k < ncol; ++k) {
        std::shuffle(permuted_positions.begin(), permuted_positions.end(), gen);
        std_dev = circular_deviations(permuted_positions, cycle_bins_, d_theta_vals);
        
        if (std_dev > EPSILON) {
            for (double& val : d_theta_vals) {
                val /= std_dev;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            D[i * ncol + k] = d_theta_vals[i];
        }
    }
    
    std::cout << "  Performing permutation test with " << PERMUTATIONS
              << " permutations in a single pass with the observed value..." << std::endl;
    std::vector<double> num;
    double W = stream_quadratic_forms(spatial_coords, D, ncol, num);
    
    if (W <= EPSILON) {
        throw std::runtime_error("Sum of weights is zero or too small");
    }
    
    std::vector<double> den(ncol, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < ncol; ++k) {
            double val = D[i * ncol + k];
            den[k] += val * val;
        }
    }
    
    if (den[0] <= EPSILON) {
        throw std::runtime_error("Denominator in Moran's I calculation is zero");
    }
    
    double observed_i = (static_cast<double>(n) / W) * (num[0] / den[0]);
    
    int total_count_greater_equal = 0;
    std::vector<double> all_permutation_values;
    all_permutation_values.reserve(PERMUTATIONS);
    
    for (size_t k = 1; k < ncol; ++k) {
        if (den[k] <= EPSILON) continue;
        
        double perm_morans_i = (static_cast<double>(n) / W) * (num[k] / den[k]);
        all_permutation_values.push_back(perm_morans_i);
        
        if (perm_morans_i >= observed_i) {
            total_count_greater_equal++;
        }
    }
    
    // Calculate empirical statistics from all permutation results
    double empirical_mean = 0.0;
    double empirical_std = 0.0;
    
    if (!all_permutation_values.empty()) {
        double sum = std::accumulate(all_permutation_values.begin(), all_permutation_values.end(), 0.0);
        empirical_mean = sum / all_permutation_values.size();
        
        double variance = 0.0;
        for (double value : all_permutation_values) {
            variance += (value - empirical_mean) * (value - empirical_mean);
        }
        variance /= all_permutation_values.size();
        empirical_std = std::sqrt(variance);
    }
    
    // Calculate p-value (one-tailed test for positive autocorrelation)
    // Add 1 to both numerator and denominator to avoid p=0 (conservative approach)
    double p_value = static_cast<double>(total_count_greater_equal + 1) / static_cast<double>(PERMUTATIONS + 1);
    
    std::cout << "  Permutation test complete: " << total_count_greater_equal 
              << "/" << PERMUTATIONS << " permutations >= observed I" << std::endl;
    std::cout << "  Empirical mean = " << empirical_mean << ", std = " << empirical_std << std::endl;
    
    bool is_significant = p_value < 0.05;
    
    return MoransIResult(observed_i, p_value, is_significant);
}