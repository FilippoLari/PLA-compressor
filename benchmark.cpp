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

template<class pla_t, typename X, typename Y>
void measure_predict_time(const std::string &dataset, const std::string &name, const uint64_t epsilon,
                             pla_t &pla, const std::vector<X> &queries) {
    Y checksum = 0;

    auto start = timer::now();

    for(const auto &query : queries) {
        checksum ^= pla.predict(query);
    }

    do_not_optimize(checksum);

    double time = std::chrono::duration_cast<nanoseconds>(timer::now() - start).count() / double(queries.size());

    std::cout << dataset << "," << epsilon << "," << name << "," << pla.size() << 
                "," << pla.bps() << "," << time << std::endl;
}

template<typename X>
std::vector<X> generate_uniform_queries(uint64_t n, X u) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<X> dist(0, u - 1);

    std::vector<X> queries(n);
    for (auto &x : queries) {
        x = dist(gen);
    }

    return queries;
}

template<typename Y>
void run_benchmark(const std::string& dataset_path, uint64_t num_queries, uint64_t epsilon) {
    bool start_with_size = !std::is_same_v<Y, uint64_t>;

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    std::vector<Y> data = read_data_binary<Y, Y>(dataset_path, start_with_size);

    std::vector<uint32_t> queries = generate_uniform_queries<uint32_t>(num_queries, data.size());

    SlopeCompressedPLA<uint32_t, Y, float, pf_mixed_float_vector> opt_slope_pfor(data, epsilon);
    measure_predict_time<decltype(opt_slope_pfor), uint32_t, Y>(dataset_name, "opt_slope_pfor",
                                                                 epsilon, opt_slope_pfor, queries);
    PlainPLA<uint32_t, Y, float> plain_pla(data, epsilon);
    measure_predict_time<decltype(plain_pla), uint32_t, Y>(dataset_name, "plain_pla",
                                                                 epsilon, plain_pla, queries);
}

int main(int argc, char *argv[]) {

    if(argc != 4) {
        std::cerr << "Usage ./benchmark <dataset_path> <epsilon> <bitwidth (32|64)>" << std::endl;
        return -1;
    }

    const std::string dataset_path = argv[1];
    const uint64_t epsilon = std::stoull(argv[2]);
    constexpr uint64_t num_queries = 15000;
    const std::string bit_width = argv[3];

    const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();

    if(bit_width == "64") {
        run_benchmark<uint64_t>(dataset_path, num_queries, epsilon);
    } else if(bit_width == "32"){
        run_benchmark<uint32_t>(dataset_path, num_queries, epsilon);
    } else {
        std::cerr << "Invalid bitwidth argument. Use 32 or 64" << std::endl;
        return -1;
    }
    
    /*auto components = cpla.components_size();
    std::cout << "dataset,epsilon,component,size" << std::endl;
    for(const auto &component : components) {
        std::cout << dataset_name << "," << epsilon << "," << component.first 
            << "," << component.second << std::endl;
    }*/
}