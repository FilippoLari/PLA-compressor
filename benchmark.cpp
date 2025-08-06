#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "slope_compressed_pla.hpp"
#include "compressed_pla.hpp"
#include "index_reader.hpp"
#include "plain_pla.hpp"

using timer = std::chrono::high_resolution_clock;
using nanoseconds = std::chrono::nanoseconds;

template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<class pla_t, typename X, typename Y>
double measure_predict_time(pla_t &pla, const std::vector<X> &queries) {
    Y checksum = 0;

    auto start = timer::now();

    for(const auto &query : queries) {
        checksum ^= pla.predict(query);
    }

    do_not_optimize(checksum);

    return std::chrono::duration_cast<nanoseconds>(timer::now() - start).count() / double(queries.size());
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

int main(int argc, char *argv[]) {

    if(argc != 3) {
        std::cerr << "Usage ./benchmark path epsilon" << std::endl;
        return -1;
    }

    const std::string file_path = argv[1];
    const size_t pos = file_path.find_last_of("/"); 
    const std::string dataset_name = (pos == std::string::npos) ? file_path : file_path.substr(pos + 1);

    const uint64_t epsilon = std::stoull(argv[2]);

    std::vector<uint32_t> data(read_data_binary<uint32_t, uint32_t>(file_path, true));

    std::vector<uint32_t> queries = generate_uniform_queries<uint32_t>(15000, data.size());

    /*SlopeCompressedPLA<uint32_t, uint32_t, float, huff_float_vector> huff_slope_pla(data, epsilon);

    std::cout << dataset_name << "," << epsilon <<  ",huff_slope_pla," << huff_slope_pla.size() << 
                "," << huff_slope_pla.bps() << "," << measure_predict_time<decltype(huff_slope_pla), uint32_t, uint32_t>(huff_slope_pla, queries) << std::endl;
*/
    SlopeCompressedPLA<uint32_t, uint32_t, float, pfor_float_vector> pfor_slope_pla(data, epsilon);

    std::cout << dataset_name << "," << epsilon <<  ",pfor_slope_pla," << pfor_slope_pla.size() << 
                "," << pfor_slope_pla.bps() << "," << measure_predict_time<decltype(pfor_slope_pla), uint32_t, uint32_t>(pfor_slope_pla, queries) << std::endl;

    /*PlainPLA<uint32_t, uint32_t, float> pla(data, epsilon);

    std::cout << dataset_name << "," << epsilon << ",pla," << pla.size() << 
                "," << pla.bps() << "," << measure_predict_time<decltype(pla), uint32_t, uint32_t>(pla, queries) << std::endl;
    */

    /*PlainPLA<uint32_t, uint32_t, float> pla(data, epsilon);

    CompressedPLA<uint32_t, uint32_t> cpla(data, epsilon);

    std::cout << dataset_name << "," << epsilon <<  ",cpla," << cpla.size() << 
                "," << cpla.bps() << "," << measure_predict_time<decltype(cpla), uint32_t, uint32_t>(cpla, queries) << std::endl;

    std::cout << dataset_name << "," << epsilon << ",pla," << pla.size() << 
                "," << pla.bps() << "," << measure_predict_time<decltype(pla), uint32_t, uint32_t>(pla, queries) << std::endl;
    */

    /*auto components = cpla.components_size();

    std::cout << "dataset,epsilon,component,size" << std::endl;
    for(const auto &component : components) {
        std::cout << dataset_name << "," << epsilon << "," << component.first 
            << "," << component.second << std::endl;
    }*/
}