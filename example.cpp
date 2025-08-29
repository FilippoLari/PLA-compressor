#include <iostream>
#include <random>
#include <vector>

#include "slope_compressed_pla.hpp"
#include "succinct_pla.hpp
#include "plain_pla.hpp"

int main(void) {
    const uint64_t epsilon = 32;

    std::vector<uint64_t> data(2000000);

    std::generate(data.begin(), data.end(), std::rand);

    std::sort(data.begin(), data.end());

    PlainPLA<uint64_t, uint64_t, float> plain_pla(data, epsilon);

    SuccinctPLA<uint64_t, uint32_t> succ_pla(data, epsilon);

    SlopeCompressedPLA<uint64_t, uint32_t, float, pfor_float_vector> slope_compr_pla(data, epsilon);

    std::cout << "PLA\t\t Space (bits)\t\t Bits per Segment (bps)" << std::endl;

    std::cout << "plain\t\t" << plain_pla.size() << "\t\t" << plain_pla.bps() << std::endl;

    std::cout << "succinct\t\t" << succ_pla.size() << "\t\t" << succ_pla.bps() << std::endl;

    std::cout << "slope compr.\t\t" << slope_compr_pla.size() << "\t\t" << slope_compr_pla.bps() << std::endl;
}