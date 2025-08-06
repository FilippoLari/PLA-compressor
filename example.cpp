#include <iostream>
#include <random>

//#include "compressed_pla.hpp"
#include "plain_pla.hpp"

#include "slope_compressed_pla.hpp"

#include "pfor_vector.hpp"

int main(void) {
    /*const uint64_t epsilon = 64;

    std::vector<uint64_t> data(200000);

    std::generate(data.begin(), data.end(), std::rand);

    std::sort(data.begin(), data.end());

    pfor_vector<uint32_t> pfv(data);*/

    for(uint8_t i = 0; i < 64; i+=8) {
        for(uint8_t j = i; j < i + 8; ++j)
            std::cout << (1UL << j) - 1UL << "UL, ";
        std::cout << std::endl;
    }

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