#include <iostream>

#include "compressed_pla.hpp"

#include "sux/bits/EliasFano.hpp"

int main(void) {
    
    std::vector<uint32_t> data(10000000);

    std::generate(data.begin(), data.end(), std::rand);
    
    std::sort(data.begin(), data.end());

    CompressedPLA<uint32_t, uint32_t, 128> cpla(data);

    std::map<std::string, size_t> components = cpla.components_size();

    std::cout << "bits per segment: " << cpla.bps() << std::endl;

    std::cout << "compression ratio: " << double(160) / cpla.bps() << std::endl; 

    for(const auto &entry : components)
        std::cout << "component: " << entry.first << " bits: " << entry.second <<
            " perc: " << (double(entry.second) / double(cpla.size())) * 100 << "%" << std::endl;
}