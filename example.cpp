#include <iostream>
#include <chrono>

//#include "compressed_pla.hpp"
#include "plain_pla.hpp"
#include "huffman_vector.hpp"

int main(void) {
    
    /*std::vector<uint64_t> data(100000);
    std::generate(data.begin(), data.end(), std::rand);

    huffman_vector<uint32_t, 128> huff_vec(data);

    std::cout << "finished" << std::endl;
    std::cout << "bpe: " << double(huff_vec.size()) / double(data.size()) << std::endl;

    for(uint64_t i = 0; i < data.size(); ++i) {
        std::cout << "got: " << huff_vec[i] << " expected: " << data[i] << std::endl;
        assert(huff_vec[i] == data[i]);
    }*/

    const uint64_t epsilon = 128;

    std::vector<uint64_t> data(10000000);

    std::generate(data.begin(), data.end(), std::rand);
    
    std::sort(data.begin(), data.end());

    PlainPLA<uint64_t, uint64_t, float> pla(data, epsilon);
    
    /*CompressedPLA<uint64_t, uint64_t> cpla(data, epsilon);

    PlainPLA<uint64_t, uint64_t, float> pla(data, epsilon);

    std::map<std::string, size_t> components = cpla.components_size();

    std::cout << "bits per segment: " << pla.bps() << std::endl;
    std::cout << "bits per segment (compr.): " << cpla.bps() << std::endl;

    std::cout << "compression ratio: " << pla.bps() / cpla.bps() << std::endl; 

    std::cout << "plain space: " << pla.size() << " compr. space: " << cpla.size() << std::endl;

    std::cout << "Component, Space (Bits), Occupancy (%)" << std::endl;

    for(const auto &entry : components)
        std::cout << entry.first << ", " << entry.second << ", " <<
                 (double(entry.second) / double(cpla.size())) * 100 << std::endl;*/
}