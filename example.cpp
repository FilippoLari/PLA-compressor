#include <iostream>
#include <chrono>

#include "compressed_pla.hpp"
#include "plain_pla.hpp"

int main(void) {

    std::vector<uint64_t> data(10000000);

    std::generate(data.begin(), data.end(), std::rand);
    
    std::sort(data.begin(), data.end());

    CompressedPLA<uint64_t, uint64_t, 128> cpla(data);

    PlainPLA<uint64_t, uint64_t, float, 128> pla(data);

    std::map<std::string, size_t> components = cpla.components_size();

    std::cout << "bits per segment: " << pla.bps() << std::endl;
    std::cout << "bits per segment (compr.): " << cpla.bps() << std::endl;

    std::cout << "compression ratio: " << pla.bps() / cpla.bps() << std::endl; 

    std::cout << "plain space: " << pla.size() << " compr. space: " << cpla.size() << std::endl;

    std::cout << "Component, Space (Bits), Occupancy (%)" << std::endl;

    for(const auto &entry : components)
        std::cout << entry.first << ", " << entry.second << ", " <<
                 (double(entry.second) / double(cpla.size())) * 100 << std::endl;
}