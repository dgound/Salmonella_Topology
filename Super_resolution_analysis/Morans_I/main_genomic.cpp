#include "morans_i_genomic.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <cmath>

struct CSVRow {
    double x, y, z;
    int timepoint, walk_id;
    
    CSVRow(double x, double y, double z, int timepoint, int walk_id) 
        : x(x), y(y), z(z), timepoint(timepoint), walk_id(walk_id) {}
};

class CSVReader {
public:
    static std::vector<CSVRow> read_csv(const std::string& filename) {
        std::vector<CSVRow> data;
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::map<std::string, int> column_indices;
        bool is_header = true;
        size_t line_count = 0;

        std::cout << "Reading CSV file: " << filename << std::endl;

        while (std::getline(file, line)) {
            line_count++;

            if (is_header) {
                column_indices = parse_header(line);
                is_header = false;

                // Verify required columns exist
                if (column_indices.find("x") == column_indices.end() ||
                    column_indices.find("y") == column_indices.end() ||
                    column_indices.find("z") == column_indices.end() ||
                    column_indices.find("time-point") == column_indices.end() ||
                    column_indices.find("Walk-IDs") == column_indices.end()) {
                    throw std::runtime_error("CSV missing required columns: x, y, z, time-point, Walk-IDs");
                }

                std::cout << "Found columns - x:" << column_indices["x"]
                          << " y:" << column_indices["y"]
                          << " z:" << column_indices["z"]
                          << " time-point:" << column_indices["time-point"]
                          << " Walk-IDs:" << column_indices["Walk-IDs"] << std::endl;
                continue;
            }

            if (line_count % 100000 == 0) {
                std::cout << "Processed " << line_count << " lines..." << std::endl;
            }

            try {
                auto row = parse_line(line, column_indices);
                // Include all Walk-IDs, timepoints 0-19 (exclude timepoint 20)
                if (row.timepoint >= 0 && row.timepoint <= 19) {
                    data.push_back(row);
                }
            } catch (const std::exception& e) {
                continue;
            }
        }

        std::cout << "Read " << data.size() << " valid data points from "
                  << line_count << " total lines" << std::endl;

        return data;
    }

private:
    // Helper function to trim whitespace and carriage returns from strings
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    static std::map<std::string, int> parse_header(const std::string& header_line) {
        std::map<std::string, int> column_map;
        std::stringstream ss(header_line);
        std::string column_name;
        int index = 0;

        while (std::getline(ss, column_name, ',')) {
            column_map[trim(column_name)] = index;  // Trim whitespace and carriage returns
            index++;
        }

        return column_map;
    }

    static CSVRow parse_line(const std::string& line, const std::map<std::string, int>& column_indices) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // Use column names to get indices
        int x_idx = column_indices.at("x");
        int y_idx = column_indices.at("y");
        int z_idx = column_indices.at("z");
        int timepoint_idx = column_indices.at("time-point");
        int walk_id_idx = column_indices.at("Walk-IDs");

        if (tokens.size() <= std::max({x_idx, y_idx, z_idx, timepoint_idx, walk_id_idx})) {
            throw std::runtime_error("Insufficient columns in CSV row");
        }

        double x = std::stod(tokens[x_idx]);
        double y = std::stod(tokens[y_idx]);
        double z = std::stod(tokens[z_idx]);
        int timepoint = std::stoi(tokens[timepoint_idx]);
        int walk_id = std::stoi(tokens[walk_id_idx]);

        return CSVRow(x, y, z, timepoint, walk_id);
    }
};

// Sample data for large datasets to make computation feasible
std::vector<size_t> stratified_sample(const std::vector<CSVRow>& data, size_t sample_size) {
    std::map<int, std::vector<size_t>> timepoint_indices;
    
    // Group indices by timepoint
    for (size_t i = 0; i < data.size(); ++i) {
        timepoint_indices[data[i].timepoint].push_back(i);
    }
    
    std::vector<size_t> sampled_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Sample proportionally from each timepoint
    size_t per_timepoint = sample_size / timepoint_indices.size();
    size_t remainder = sample_size % timepoint_indices.size();
    
    for (auto& [tp, indices] : timepoint_indices) {
        size_t to_sample = per_timepoint + (remainder > 0 ? 1 : 0);
        if (remainder > 0) remainder--;
        
        to_sample = std::min(to_sample, indices.size());
        
        std::shuffle(indices.begin(), indices.end(), gen);
        for (size_t i = 0; i < to_sample; ++i) {
            sampled_indices.push_back(indices[i]);
        }
    }
    
    std::sort(sampled_indices.begin(), sampled_indices.end());
    return sampled_indices;
}

int main(int argc, char* argv[]) {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Default values
        std::string input_filename = "datasets/TestData_JA_2walks.csv";
        size_t max_sample_points = 50000; // Default 2500 points per timepoint × 20 timepoints
        
        // Parse command line arguments
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            std::cout << "Usage: " << argv[0] << " [input_file] [max_sample_points]" << std::endl;
            std::cout << "  input_file: CSV file to process (default: " << input_filename << ")" << std::endl;
            std::cout << "  max_sample_points: Maximum points per Walk-ID (default: " << max_sample_points << ")" << std::endl;
            std::cout << "\nExample: " << argv[0] << " data.csv 10000" << std::endl;
            return 0;
        }
        
        if (argc >= 2) {
            input_filename = argv[1];
        }
        if (argc >= 3) {
            try {
                max_sample_points = std::stoul(argv[2]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid sample size '" << argv[2] << "'. Must be a positive integer." << std::endl;
                return 1;
            }
        }
        
        std::cout << "Input file: " << input_filename << std::endl;
        std::cout << "Max sample points per Walk-ID: " << max_sample_points << std::endl;
        
        // Read CSV data
        auto csv_data = CSVReader::read_csv(input_filename);
        
        // Get basename of input file for output
        std::string basename = std::filesystem::path(input_filename).stem().string();
        
        // Group by Walk-ID
        std::map<int, std::vector<CSVRow>> walk_groups;
        for (const auto& row : csv_data) {
            walk_groups[row.walk_id].push_back(row);
        }
        
        std::cout << "\nFound " << walk_groups.size() << " unique Walk-IDs" << std::endl;
        
        // Create genomic Moran's I calculator
        GenomicMoransICalculator calculator;
        
        // Prepare CSV output file
        std::string output_filename = basename + "_morans_i_results.csv";
        std::ofstream output_file(output_filename);
        if (!output_file.is_open()) {
            throw std::runtime_error("Could not create output file: " + output_filename);
        }
        
        // Write CSV header
        output_file << "Filename,Walk-ID,Time-points,I-value,P-value\n";
        
        // Store results for CSV output
        struct ResultRow {
            std::string filename;
            int walk_id;
            int time_points;
            double i_value;
            double p_value;
        };
        std::vector<ResultRow> results;
        
        // Process each Walk-ID
        for (auto& [walk_id, rows] : walk_groups) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Processing Walk-ID: " << walk_id << std::endl;
            std::cout << "Total data points: " << rows.size() << std::endl;
            
            // For large datasets, use stratified sampling to maintain timepoint representation
            std::vector<size_t> sample_indices;
            
            if (rows.size() > max_sample_points) {
                std::cout << "Dataset too large, using stratified sampling of " << max_sample_points << " points..." << std::endl;
                sample_indices = stratified_sample(rows, max_sample_points);
                std::cout << "Selected " << sample_indices.size() << " representative points" << std::endl;
            } else {
                // Use all data points
                sample_indices.resize(rows.size());
                std::iota(sample_indices.begin(), sample_indices.end(), 0);
            }
            
            // Extract sampled data
            std::vector<Point3D> points;
            std::vector<int> timepoints;
            
            points.reserve(sample_indices.size());
            timepoints.reserve(sample_indices.size());
            
            for (size_t idx : sample_indices) {
                const auto& row = rows[idx];
                points.emplace_back(row.x, row.y, row.z);
                timepoints.push_back(row.timepoint);
            }
            
            // Show timepoint distribution and count unique timepoints
            std::set<int> unique_timepoints;
            std::map<int, int> tp_counts;
            for (int tp : timepoints) {
                unique_timepoints.insert(tp);
                tp_counts[tp]++;
            }
            int actual_timepoints = unique_timepoints.size();
            
            std::cout << "Timepoint distribution: ";
            for (auto& [tp, count] : tp_counts) {
                std::cout << tp << ":" << count << " ";
            }
            std::cout << std::endl;
            std::cout << "Actual timepoints present: " << actual_timepoints << std::endl;
            
            try {
                auto walk_start = std::chrono::high_resolution_clock::now();
                
                // Compute genomic Moran's I with p-value (spatial weights, circular genomic variable)
                MoransIResult result = calculator.compute_genomic_morans_i_with_pvalue(points, timepoints);
                
                auto walk_end = std::chrono::high_resolution_clock::now();
                auto walk_duration = std::chrono::duration_cast<std::chrono::seconds>(walk_end - walk_start);
                
                std::cout << "\n" << std::string(50, '-') << std::endl;
                std::cout << "RESULTS FOR Walk-ID " << walk_id << ":" << std::endl;
                std::cout << "Genomic Moran's I = " << std::fixed << std::setprecision(6) << result.morans_i << std::endl;
                std::cout << "P-value = " << std::fixed << std::setprecision(6) << result.p_value << std::endl;
                std::cout << "Statistically significant: " << (result.is_significant ? "YES" : "NO") << " (α = 0.05)" << std::endl;
                std::cout << "Computation time: " << walk_duration.count() << " seconds" << std::endl;
                
                // Statistical interpretation
                if (result.is_significant) {
                    if (result.morans_i > 0) {
                        std::cout << "INTERPRETATION: Statistically significant POSITIVE spatial-genomic correlation" << std::endl;
                        std::cout << "                Spatially close points tend to be genomically adjacent" << std::endl; 
                        std::cout << "                ✅ Evidence of circular genome organization!" << std::endl;
                    } else {
                        std::cout << "INTERPRETATION: Statistically significant NEGATIVE spatial-genomic correlation" << std::endl;
                        std::cout << "                Spatially close points tend to be genomically distant" << std::endl;
                        std::cout << "                ❌ Evidence against circular organization" << std::endl;
                    }
                } else {
                    std::cout << "INTERPRETATION: NO statistically significant spatial-genomic correlation" << std::endl;
                    std::cout << "                Pattern is not significantly different from random" << std::endl;
                    std::cout << "                ⚠️  Cannot conclude circular organization exists" << std::endl;
                }
                std::cout << std::string(50, '-') << std::endl;
                
                // Store result for CSV output
                results.push_back({basename, walk_id, actual_timepoints, result.morans_i, result.p_value});
                
            } catch (const std::exception& e) {
                std::cerr << "Error computing genomic Moran's I for Walk-ID " << walk_id 
                          << ": " << e.what() << std::endl;
                // Store failed result as NaN
                results.push_back({basename, walk_id, actual_timepoints, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()});
            }
        }
        
        // Write results to CSV
        for (const auto& result : results) {
            output_file << result.filename << "," 
                       << result.walk_id << ","
                       << result.time_points << ",";
            
            if (std::isnan(result.i_value)) {
                output_file << "NaN";
            } else {
                output_file << std::fixed << std::setprecision(6) << result.i_value;
            }
            
            output_file << ",";
            
            if (std::isnan(result.p_value)) {
                output_file << "NaN";
            } else {
                output_file << std::fixed << std::setprecision(6) << result.p_value;
            }
            
            output_file << "\n";
        }
        
        output_file.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "GENOMIC ANALYSIS COMPLETED" << std::endl;
        std::cout << "Total execution time: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Results saved to: " << output_filename << std::endl;
        std::cout << "\nThis analysis tests: 'Do spatially close points have similar genomic positions?'" << std::endl;
        std::cout << "Positive values indicate your OligoSTORM captured circular genome organization." << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}