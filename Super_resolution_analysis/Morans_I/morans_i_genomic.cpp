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

double GenomicMoransICalculator::compute_genomic_morans_i(
    const std::vector<Point3D>& spatial_coords,
    const std::vector<int>& genomic_positions) const {
    
    validate_input(spatial_coords, genomic_positions);
    
    const size_t n = spatial_coords.size();
    std::cout << "Computing genomic Moran's I for " << n << " points..." << std::endl;
    
    // Step 1: Compute spatial inverse distance weights (like Python pdist)
    std::cout << "Computing spatial distance weights..." << std::endl;
    std::vector<std::vector<double>> weights(n, std::vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = spatial_coords[i].distance_to(spatial_coords[j]);
            double weight = 1.0 / (dist * dist + EPSILON);
            weights[i][j] = weight;
            weights[j][i] = weight; // Symmetric
        }
        weights[i][i] = 0.0; // Diagonal is zero
    }
    
    // Step 2: Row standardize weights
    std::cout << "Row standardizing weights..." << std::endl;
    row_standardize_weights(weights);
    
    // Step 3: Compute sum of weights
    double W = 0.0;
    for (const auto& row : weights) {
        W += std::accumulate(row.begin(), row.end(), 0.0);
    }
    
    if (W <= EPSILON) {
        throw std::runtime_error("Sum of weights is zero or too small");
    }
    
    // Step 4: Compute circular genomic variable (EXACTLY like Python)
    std::cout << "Computing circular genomic deviations..." << std::endl;
    
    // Map timepoints to angles: angles = X * 2 * π / 20
    std::vector<double> angles(n);
    for (size_t i = 0; i < n; ++i) {
        angles[i] = genomic_positions[i] * 2.0 * M_PI / 20.0;
    }
    
    // Compute circular mean (like Python)
    double sum_cos = 0.0, sum_sin = 0.0;
    for (double angle : angles) {
        sum_cos += std::cos(angle);
        sum_sin += std::sin(angle);
    }
    double mean_cos = sum_cos / n;
    double mean_sin = sum_sin / n;
    double X_bar = std::atan2(mean_sin, mean_cos);
    
    // Compute circular deviations (EXACTLY like Python)
    std::vector<double> d_theta_vals(n);
    for (size_t i = 0; i < n; ++i) {
        double diff = angles[i] - X_bar;
        d_theta_vals[i] = std::atan2(std::sin(diff), std::cos(diff));
    }
    
    // Standardize deviations (like Python)
    double std_dev = 0.0;
    for (double val : d_theta_vals) {
        std_dev += val * val;
    }
    std_dev = std::sqrt(std_dev / n);
    
    if (std_dev > EPSILON) {
        for (double& val : d_theta_vals) {
            val /= std_dev;
        }
    } else {
        throw std::runtime_error("Standard deviation of circular deviations is zero");
    }
    
    // Step 5: Compute Moran's I (like Python)
    std::cout << "Computing final Moran's I value..." << std::endl;
    
    // numerator = d_theta_vals @ weights @ d_theta_vals
    double numerator = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            numerator += d_theta_vals[i] * weights[i][j] * d_theta_vals[j];
        }
    }
    
    // denominator = sum(d_theta_vals^2)
    double denominator = 0.0;
    for (double val : d_theta_vals) {
        denominator += val * val;
    }
    
    if (denominator <= EPSILON) {
        throw std::runtime_error("Denominator in Moran's I calculation is zero");
    }
    
    // I = (N / W) * (numerator / denominator)
    double morans_i = (static_cast<double>(n) / W) * (numerator / denominator);
    return morans_i;
}

void GenomicMoransICalculator::row_standardize_weights(std::vector<std::vector<double>>& weights) const {
    const size_t n = weights.size();
    
    for (size_t i = 0; i < n; ++i) {
        double row_sum = std::accumulate(weights[i].begin(), weights[i].end(), 0.0);
        
        if (row_sum > EPSILON) {
            for (size_t j = 0; j < n; ++j) {
                weights[i][j] /= row_sum;
            }
        }
    }
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

// Result structure for permutation worker
struct PermutationWorkerResult {
    int count_greater_equal;
    std::vector<double> morans_i_values;
    
    PermutationWorkerResult() : count_greater_equal(0) {}
};

// Helper function for multithreaded permutation worker
PermutationWorkerResult permutation_worker(
    const std::vector<Point3D>& spatial_coords,
    const std::vector<int>& genomic_positions,
    const std::vector<std::vector<double>>& weights,
    double observed_morans_i,
    int num_permutations,
    int thread_id) {
    
    const size_t n = spatial_coords.size();
    const double EPSILON = 1e-10;
    
    // Each thread gets its own random generator seeded differently
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    
    PermutationWorkerResult result;
    result.morans_i_values.reserve(num_permutations);
    std::vector<int> permuted_positions = genomic_positions;
    
    for (int perm = 0; perm < num_permutations; ++perm) {
        // Randomly permute the genomic positions
        std::shuffle(permuted_positions.begin(), permuted_positions.end(), gen);
        
        // Convert to circular coordinates and standardize
        std::vector<double> angles(n);
        for (size_t i = 0; i < n; ++i) {
            angles[i] = permuted_positions[i] * 2.0 * M_PI / 20.0;
        }
        
        // Compute circular mean
        double sum_cos = 0.0, sum_sin = 0.0;
        for (double angle : angles) {
            sum_cos += std::cos(angle);
            sum_sin += std::sin(angle);
        }
        double mean_cos = sum_cos / n;
        double mean_sin = sum_sin / n;
        double X_bar = std::atan2(mean_sin, mean_cos);
        
        // Compute circular deviations
        std::vector<double> d_theta_vals(n);
        for (size_t i = 0; i < n; ++i) {
            double diff = angles[i] - X_bar;
            d_theta_vals[i] = std::atan2(std::sin(diff), std::cos(diff));
        }
        
        // Standardize deviations
        double std_dev = 0.0;
        for (double val : d_theta_vals) {
            std_dev += val * val;
        }
        std_dev = std::sqrt(std_dev / n);
        
        if (std_dev > EPSILON) {
            for (double& val : d_theta_vals) {
                val /= std_dev;
            }
        }
        
        // Calculate Moran's I for this permutation
        try {
            // Inline core calculation to avoid creating calculator instance
            double W = 0.0;
            for (const auto& row : weights) {
                W += std::accumulate(row.begin(), row.end(), 0.0);
            }
            
            if (W <= EPSILON) continue;
            
            double numerator = 0.0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    numerator += d_theta_vals[i] * weights[i][j] * d_theta_vals[j];
                }
            }
            
            double denominator = 0.0;
            for (double val : d_theta_vals) {
                denominator += val * val;
            }
            
            if (denominator <= EPSILON) continue;
            
            double perm_morans_i = (static_cast<double>(n) / W) * (numerator / denominator);
            
            // Store the result
            result.morans_i_values.push_back(perm_morans_i);
            
            if (perm_morans_i >= observed_morans_i) {
                result.count_greater_equal++;
            }
        } catch (const std::exception& e) {
            // Skip this permutation if calculation fails
            continue;
        }
    }
    
    return result;
}

GenomicMoransICalculator::PermutationTestResult GenomicMoransICalculator::calculate_permutation_test(
    const std::vector<Point3D>& spatial_coords,
    const std::vector<int>& genomic_positions,
    double observed_morans_i) const {
    
    const size_t n = spatial_coords.size();
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "  Detected " << num_threads << " CPU threads available" << std::endl;
    //const unsigned int threads_to_use = std::min(num_threads, static_cast<unsigned int>(8)); // Cap at 8 threads
    const unsigned int threads_to_use = num_threads;
    
    std::cout << "  Performing permutation test with " << PERMUTATIONS << " permutations using " 
              << threads_to_use << " threads..." << std::endl;
    
    // Create spatial weights matrix (shared by all threads)
    std::vector<std::vector<double>> weights(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = spatial_coords[i].distance_to(spatial_coords[j]);
            double weight = 1.0 / (dist * dist + EPSILON);
            weights[i][j] = weight;
            weights[j][i] = weight;
        }
        weights[i][i] = 0.0;
    }
    row_standardize_weights(weights);
    
    // Distribute permutations across threads
    std::vector<std::future<PermutationWorkerResult>> futures;
    int perms_per_thread = PERMUTATIONS / threads_to_use;
    int remaining_perms = PERMUTATIONS % threads_to_use;
    
    for (unsigned int t = 0; t < threads_to_use; ++t) {
        int thread_perms = perms_per_thread + (t < remaining_perms ? 1 : 0);
        
        futures.push_back(std::async(std::launch::async, permutation_worker,
            std::cref(spatial_coords),
            std::cref(genomic_positions), 
            std::cref(weights),
            observed_morans_i,
            thread_perms,
            static_cast<int>(t)));
    }
    
    // Collect results from all threads
    int total_count_greater_equal = 0;
    std::vector<double> all_permutation_values;
    
    for (auto& future : futures) {
        PermutationWorkerResult worker_result = future.get();
        total_count_greater_equal += worker_result.count_greater_equal;
        
        // Combine all Moran's I values from all threads
        all_permutation_values.insert(all_permutation_values.end(),
            worker_result.morans_i_values.begin(),
            worker_result.morans_i_values.end());
    }
    
    // Calculate empirical statistics from all permutation results
    double empirical_mean = 0.0;
    double empirical_std = 0.0;
    
    if (!all_permutation_values.empty()) {
        // Calculate mean
        double sum = std::accumulate(all_permutation_values.begin(), all_permutation_values.end(), 0.0);
        empirical_mean = sum / all_permutation_values.size();
        
        // Calculate standard deviation
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
    
    return PermutationTestResult(p_value, empirical_mean, empirical_std, all_permutation_values.size());
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
    const std::vector<int>& genomic_positions) const {
    
    validate_input(spatial_coords, genomic_positions);
    
    std::cout << "Computing genomic Moran's I with p-value..." << std::endl;
    
    const size_t n = spatial_coords.size();
    
    // First compute the observed Moran's I
    double observed_i = compute_genomic_morans_i(spatial_coords, genomic_positions);
    
    // Compute weights matrix for z-score calculation
    std::cout << "Computing variance for z-score..." << std::endl;
    std::vector<std::vector<double>> weights(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = spatial_coords[i].distance_to(spatial_coords[j]);
            double weight = 1.0 / (dist * dist + EPSILON);
            weights[i][j] = weight;
            weights[j][i] = weight;
        }
        weights[i][i] = 0.0;
    }
    row_standardize_weights(weights);
    
    // Compute permutation test for p-value
    PermutationTestResult perm_result = calculate_permutation_test(spatial_coords, genomic_positions, observed_i);
    
    bool is_significant = perm_result.p_value < 0.05;
    
    return MoransIResult(observed_i, perm_result.p_value, is_significant);
}