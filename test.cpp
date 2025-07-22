#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <utility>
#include <random>

#include "slope_compressed_pla.hpp"
#include "compressed_pla.hpp"
#include "plain_pla.hpp"

#include "huffman_vector.hpp"
#include "dist_vector.hpp"
#include "rle_vector.hpp"
#include "float_vector.hpp"

class TestingData : public ::testing::Test {
protected:

    static std::vector<uint32_t> int_data;

    static std::vector<uint32_t> sorted_int_data;

    static std::vector<float> float_data;

    static constexpr uint32_t size = (1 << 20);

    static constexpr uint32_t seed = 42;

    template <typename T>
    static void init_data(std::vector<T>& data, T min, T max) {
        data = std::vector<T>(size);
        std::mt19937 gen(seed);
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dis(min, max);
            for (uint64_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
        } else if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(min, max);
            for (uint64_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
        }
    }

    static void SetUpTestSuite() {
        if(int_data.empty() && float_data.empty()) {
            init_data<uint32_t>(int_data, std::numeric_limits<uint32_t>::min(),
                            std::numeric_limits<uint32_t>::max());
            sorted_int_data.assign(int_data.begin(), int_data.end());
            std::sort(sorted_int_data.begin(), sorted_int_data.end());
            init_data<float>(float_data, std::numeric_limits<float>::min(),
                            std::numeric_limits<float>::max()); 
        }
    }
};

std::vector<uint32_t> TestingData::int_data;
std::vector<uint32_t> TestingData::sorted_int_data;
std::vector<float> TestingData::float_data;

TEST_F(TestingData, HuffVectorTest) {
    huffman_vector<uint32_t, 64> hv(int_data);
    for(uint64_t i = 0; i < int_data.size(); ++i) {
        ASSERT_EQ(hv[i], int_data[i]) << " hv[i] = " << hv[i] << ", data[i] = " << int_data[i] << 
                                        " i: " << i;
    }
}

TEST_F(TestingData, DistVectorTest) {
    dist_vector<uint32_t> dv(int_data);
    for(uint64_t i = 0; i < int_data.size(); ++i) {
        ASSERT_EQ(dv[i], int_data[i]) << " dv[i] = " << dv[i] << ", data[i] = " << int_data[i] << 
                                        " i: " << i;
    }
}

TEST_F(TestingData, HuffFloatVectorTest) {
    huff_float_vector hfv(float_data);
    for(uint64_t i = 0; i < float_data.size(); ++i) {
        ASSERT_EQ(hfv[i], float_data[i]) << " hfv[i] = " << hfv[i] << ", data[i] = " << float_data[i] << 
                                        " i: " << i;
    }
}

TEST_F(TestingData, DistFloatVectorTest) {
    dist_float_vector dfv(float_data);
    for(uint64_t i = 0; i < float_data.size(); ++i) {
        ASSERT_EQ(dfv[i], float_data[i]) << " dfv[i] = " << dfv[i] << ", data[i] = " << float_data[i] << 
                                        " i: " << i;
    }
}

TEST_F(TestingData, RunLengthVectorTest) {
    rle_vector<uint32_t> rlev(int_data);
    for(uint64_t i = 0; i < int_data.size(); ++i) {
        ASSERT_EQ(rlev[i], int_data[i]) << " rlev[i] = " << rlev[i] << ", data[i] = " << int_data[i] << 
                                        " i: " << i;
    }
}

TEST_F(TestingData, PlainPLATest) {
    PlainPLA<uint32_t, uint32_t, float> pla(sorted_int_data, 128);
    const int32_t range_size = 2*(128 + 2);
    for(uint64_t i = 0; i < sorted_int_data.size(); ++i) {
        const uint32_t y = sorted_int_data[i];
        const uint32_t pred = pla.predict(i);
        int32_t diff = int32_t(pred) - int32_t(y);
        diff = (diff < 0) ? -diff : diff;
        ASSERT_TRUE((diff <= range_size)) << "pred: " << pred << " real: " << y << " diff: " << diff;
    }
}

TEST_F(TestingData, SuccinctPLATest) {
    CompressedPLA<uint32_t, uint32_t> cpla(sorted_int_data, 128);
    const int32_t range_size = 2*(128 + 3);
    for(uint64_t i = 0; i < sorted_int_data.size(); ++i) {
        const uint32_t y = sorted_int_data[i];
        int32_t diff = int32_t(cpla.predict(i)) - int32_t(y);
        diff = (diff < 0) ? (-1) * diff : diff;
        ASSERT_TRUE((diff <= range_size)) << "diff: " << diff << " expected: " << range_size;
    }
}

TEST_F(TestingData, SlopeCompressedPLATest) {
    SlopeCompressedPLA<uint32_t, uint32_t, float, dist_float_vector> scpla(sorted_int_data, 128);
    const int32_t range_size = 2*(128 + 2);
    for(uint64_t i = 0; i < sorted_int_data.size(); ++i) {
        const uint32_t y = sorted_int_data[i];
        const uint32_t pred = scpla.predict(i);
        int32_t diff = int32_t(pred) - int32_t(y);
        diff = (diff < 0) ? -diff : diff;
        ASSERT_TRUE((diff <= range_size)) << "pos: " << i << " pred: " << pred << " real: " << y << " diff: " << diff;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}