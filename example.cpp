#include <iostream>
#include <random>

#include "plain_pla.hpp"
#include "slope_compressed_pla.hpp"

int main(void) {
    const uint64_t epsilon = 64;

    std::vector<uint64_t> data(200000);

    std::generate(data.begin(), data.end(), std::rand);

    std::sort(data.begin(), data.end());

    /*SuccinctPLA<uint64_t, uint32_t> spla(data, epsilon);

    std::cout << "bps: " << spla.bps() << std::endl;

    std::cout << "Component, Space (Bits), Occupancy (%)" << std::endl;

    for(const auto &entry : spla.components_size())
        std::cout << entry.first << ", " << entry.second << ", " <<
                 (double(entry.second) / double(spla.size())) * 100 << std::endl;*/

    SlopeCompressedPLA<uint32_t, uint64_t, float, dist_float_vector> scpla(data, epsilon);

    /*PlainPLA<uint64_t, uint64_t, float> pla(data, epsilon);

    std::cout << pla.bps() << std::endl;

    std::cout << pla.bps_lower_bound(data) << std::endl;*

    //SlopeCompressedPLA<uint64_t, uint64_t, float, huff_float_vector> pla(data, epsilon);
    
    //std::cout << "bps: " << pla.bps() << std::endl;

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