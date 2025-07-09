#include <iostream>

#include "compressed_pla.hpp"

int main(void) {
    
    std::vector<uint32_t> data(10000000);

    std::generate(data.begin(), data.end(), std::rand);
    
    std::sort(data.begin(), data.end());

    std::vector<uint32_t> unique_data;

    for(uint32_t i = 1; i < data.size(); ++i)
        if(data[i] != data[i - 1]) unique_data.push_back(data[i]); 

    CompressedPLA<uint32_t, uint32_t, 128> cpla(unique_data);

    std::cout << "bps: " << cpla.bps() << std::endl;
}