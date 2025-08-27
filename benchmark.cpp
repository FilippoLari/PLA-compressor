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
std::vector<T> generate_uniform_queries(size_t n, T min, T max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(min, max);

    std::vector<T> queries(n);
    for (auto &x : queries) {
        x = dist(gen);
    }

    return queries;
}

template<typename T, bool Indexing>
void run_benchmark(const std::string& dataset_path, uint64_t num_queries, uint64_t epsilon) {
    bool start_with_size = !Indexing;

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    std::vector<T> data = read_data_binary<T, T>(dataset_path, start_with_size);
    // some datasets used for the indexing experiments are not sorted
    std::sort(data.begin(), data.end()); 

    const T min = (Indexing) ? data[0] : 0;
    const T max = (Indexing) ? data.back() : data.size() - 1;

    std::vector<T> queries = generate_uniform_queries<T>(num_queries, min, max);

    PlainPLA<T, T, float, Indexing> plain_pla(data, epsilon);
    measure_predict_time<decltype(plain_pla), T, Indexing>(dataset_name, "plain_pla",
                                                        epsilon, plain_pla, queries);

    SlopeCompressedPLA<T, T, float, pf_mixed_float_vector, Indexing> opt_slope_pfor(data, epsilon);
    measure_predict_time<decltype(opt_slope_pfor), T, Indexing>(dataset_name, "opt_slope_pfor",
                                                        epsilon, opt_slope_pfor, queries);

    SlopeCompressedPLA<T, T, float, huff_mixed_float_vector, Indexing> opt_slope_huff(data, epsilon);
    measure_predict_time<decltype(opt_slope_huff), T, Indexing>(dataset_name, "opt_slope_huff",
                                                        epsilon, opt_slope_huff, queries);

    SuccinctPLA<T, T, Indexing> succinct_pla(data, epsilon);
    measure_predict_time<decltype(succinct_pla), T, Indexing>(dataset_name, "succinct_pla",
                                                        epsilon, succinct_pla, queries);
}

int main(int argc, char *argv[]) {

    if(argc != 4) {
        std::cerr << "Usage ./benchmark <dataset_path> <epsilon> <mode (indexing|compression)>" << std::endl;
        return -1;
    }

    const std::string dataset_path = argv[1];
    const uint64_t epsilon = std::stoull(argv[2]);
    constexpr uint64_t num_queries = 15000;
    const std::string mode = argv[3];

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    if(mode == "indexing") {
        run_benchmark<uint64_t, true>(dataset_path, num_queries, epsilon);
    } else if(mode == "compression"){
        run_benchmark<uint32_t, true>(dataset_path, num_queries, epsilon);
    } else {
        std::cerr << "Invalid mode. Use 'indexing' or 'compression'" << std::endl;
        return -1;
    }

    return 0;
}