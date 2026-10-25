#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

struct Point3D {
    double x, y, z;
    Point3D(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    double distance_to(const Point3D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

struct MoransIResult {
    double morans_i;
    double p_value;
    bool is_significant;
    
    MoransIResult(double i, double p, bool sig) 
        : morans_i(i), p_value(p), is_significant(sig) {}
};

class GenomicMoransICalculator {
private:
    static constexpr double EPSILON = 1e-10;
    static constexpr int TOTAL_CHUNKS = 20;
    static constexpr int CHUNK_SIZE_BP = 250000;
    static constexpr int PERMUTATIONS = 100;

    // Number of genomic bins on the circle: timepoint tp maps to angle tp * 2pi / cycle_bins_
    double cycle_bins_;

    // Compute circular genomic distance between two chunks
    double genomic_distance(int chunk1, int chunk2) const;
    
    // Compute genomic-based inverse distance weights
    std::vector<std::vector<double>> compute_genomic_weights(
        const std::vector<int>& genomic_positions) const;
    
    // Compute spatial statistics per genomic chunk
    std::vector<double> compute_spatial_variable(
        const std::vector<Point3D>& spatial_coords,
        const std::vector<int>& genomic_positions) const;
    
    // Standardize variable values
    void standardize_variable(std::vector<double>& variable) const;
    
    // Core Moran's I calculation (without p-value)
    double calculate_morans_i_core(
        const std::vector<std::vector<double>>& weights,
        const std::vector<double>& standardized_variable) const;
    
    // Streams row-standardized inverse squared distance weights once and returns
    // num_out[k] = D[:,k]^T W D[:,k] for each of the ncol columns of row-major D; returns sum of weights
    double stream_quadratic_forms(
        const std::vector<Point3D>& spatial_coords,
        const std::vector<double>& D,
        size_t ncol,
        std::vector<double>& num_out) const;

    // Calculate variance of Moran's I under null hypothesis
    double calculate_morans_i_variance(
        const std::vector<std::vector<double>>& weights) const;

public:
    explicit GenomicMoransICalculator(double cycle_bins = 20.0) : cycle_bins_(cycle_bins) {}
    ~GenomicMoransICalculator() = default;
    
    // Main computation with p-value: genomic weights, spatial variable.
    // A nonzero seed makes the permutation shuffles reproducible; 0 uses std::random_device.
    MoransIResult compute_genomic_morans_i_with_pvalue(
        const std::vector<Point3D>& spatial_coords,
        const std::vector<int>& genomic_positions,
        unsigned long long seed = 0) const;
    
    // Legacy method (without p-value for backward compatibility)
    double compute_genomic_morans_i(const std::vector<Point3D>& spatial_coords,
                                   const std::vector<int>& genomic_positions) const;
    
    // Validate input
    void validate_input(const std::vector<Point3D>& spatial_coords,
                       const std::vector<int>& genomic_positions) const;
};