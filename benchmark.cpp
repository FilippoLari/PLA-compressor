#include <type_traits>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "slope_compressed_pla.hpp"
#include "succinct_pla.hpp"
#include "index_reader.hpp"
#include "experiments.hpp"
#include "plain_pla.hpp"

using timer = std::chrono::high_resolution_clock;
using nanoseconds = std::chrono::nanoseconds;

template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<class pla_t, typename X, bool Indexing>
void measure_predict_time(const std::string &dataset, const std::string &name, const uint64_t epsilon,
                             pla_t &pla, const std::vector<X> &queries) {
    int64_t checksum = 0;

    auto start = timer::now();

    for(const auto &query : queries) {
        checksum ^= pla.predict(query);
    }

    do_not_optimize(checksum);

    double time = std::chrono::duration_cast<nanoseconds>(timer::now() - start).count() / double(queries.size());

    const std::string mode = (Indexing) ? "indexing" : "compression";

    std::cout << dataset << "," << mode << "," << epsilon << "," << name << "," << pla.size() << 
                "," << pla.bps() << "," << time << std::endl;
}

template<typename T>
std::vector<T> generate_spaced_queries(size_t n, T min, T max) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<T> queries(n);

    T interval = (max - min) / n;

    for (size_t i = 0; i < n; ++i) {
        std::uniform_int_distribution<T> dist(min + i * interval, min + (i + 1) * interval - 1);
        queries[i] = dist(gen);
    }

    return queries;
}

template<typename T, bool Indexing>
void run_benchmark(const std::string& dataset_path, uint64_t num_queries, uint64_t epsilon) {
    bool start_with_size = !Indexing;

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    std::vector<T> data = read_data_binary<T, T>(dataset_path, start_with_size);
    // some datasets used for the indexing experiments are not sorted (??)
    std::sort(data.begin(), data.end()); 

    const T min = (Indexing) ? data[0] : 0;
    const T max = (Indexing) ? data.back() : data.size() - 1;

    std::vector<T> queries = generate_spaced_queries<T>(num_queries, min, max);

    PlainPLA<T, T, float, Indexing> plain_pla(data, epsilon);
    measure_predict_time<decltype(plain_pla), T, Indexing>(dataset_name, "plain_pla",
                                                        epsilon, plain_pla, queries);

    if constexpr (Indexing) 
        std::cout << dataset_name << "," << "indexing" << "," << epsilon << "," <<
                 "lower_bound" << "," << plain_pla.lower_bound_indexing() << 
                    "," << plain_pla.lower_bound_indexing() / double(plain_pla.get_segments()) <<
                     ",0" << std::endl;
    else
        std::cout << dataset_name << "," << "compression" << "," << epsilon << "," <<
                 "lower_bound" << "," << plain_pla.lower_bound_compression(data) << 
                    "," << plain_pla.lower_bound_compression(data) / double(plain_pla.get_segments()) <<
                     ",0" << std::endl;

    SlopeCompressedPLA<T, T, float, pfor_float_vector, Indexing> opt_slope_pfor(data, epsilon);
    measure_predict_time<decltype(opt_slope_pfor), T, Indexing>(dataset_name, "opt_slope_pfor",
                                                        epsilon, opt_slope_pfor, queries);

    SlopeCompressedPLA<T, T, float, huff_float_vector, Indexing> opt_slope_huff(data, epsilon);
    measure_predict_time<decltype(opt_slope_huff), T, Indexing>(dataset_name, "opt_slope_huff",
                                                        epsilon, opt_slope_huff, queries);

    SuccinctPLA<T, T, Indexing> succinct_pla(data, epsilon);
    measure_predict_time<decltype(succinct_pla), T, Indexing>(dataset_name, "succinct_pla",
                                                        epsilon, succinct_pla, queries);
}

template<typename T, bool Indexing>
void measure_entropy(const std::string& dataset_path, uint64_t epsilon) {
    bool start_with_size = !Indexing;

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    std::vector<T> data = read_data_binary<T, T>(dataset_path, start_with_size);
    // some datasets used for the indexing experiments are not sorted (??)
    std::sort(data.begin(), data.end()); 

    const auto [signs, exponents, mantissae, 
                opt_signs, opt_exponents, opt_mantissae] = slope_components_entropy<T, T, float, Indexing>(data, epsilon);

    const std::string mode = (Indexing) ? "indexing" : "compression";

    std::cout << dataset_name << "," << mode << "," << epsilon << "," <<
                 signs << "," << exponents << "," << mantissae << "," <<
                 opt_signs << "," << opt_exponents << "," << opt_mantissae << std::endl;
}

template<typename T, bool Indexing>
void print_frequencies(const std::string& dataset_path, uint64_t epsilon) {
    bool start_with_size = !Indexing;

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    std::vector<T> data = read_data_binary<T, T>(dataset_path, start_with_size);
    // some datasets used for the indexing experiments are not sorted (??)
    std::sort(data.begin(), data.end()); 

    const auto [mant_freq, opt_mant_freq] = mantissae_frequencies<T, T, float, Indexing>(data, epsilon);
                
    for(const auto &[element, frequency] : mant_freq) 
        std::cout << dataset_name << "," << epsilon << ",original," << element << "," << frequency << std::endl;

    for(const auto &[element, frequency] : opt_mant_freq)
        std::cout << dataset_name << "," << epsilon << ",optimized," << element << "," << frequency << std::endl;
}

int main(int argc, char *argv[]) {

    if(argc != 5) {
        std::cerr << "Usage ./benchmark <dataset_path> <epsilon> <mode (indexing|compression)> <run (benchmark|entropy|frequency)>" << std::endl;
        return -1;
    }

    const std::string dataset_path = argv[1];
    const uint64_t epsilon = std::stoull(argv[2]);
    constexpr uint64_t num_queries = 15000;
    const std::string mode = argv[3];
    const std::string run = argv[4];

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    const bool is_indexing = (mode == "indexing");

    if (run == "benchmark") {
        if (is_indexing)
            run_benchmark<uint64_t, true>(dataset_path, num_queries, epsilon);
        else
            run_benchmark<uint32_t, false>(dataset_path, num_queries, epsilon);
    } else if (run == "entropy") {
        if (is_indexing)
            measure_entropy<uint64_t, true>(dataset_path, epsilon);
        else
            measure_entropy<uint32_t, false>(dataset_path, epsilon);
    } else if (run == "frequency") {
        if (is_indexing)
            print_frequencies<uint64_t, true>(dataset_path, epsilon);
        else
            print_frequencies<uint32_t, false>(dataset_path, epsilon);
    } else {
        std::cerr << "Invalid parameter run. Choose between 'benchmark', 'entropy' or 'frequency'" << std::endl;
        return -1;
    }

    return 0;
}