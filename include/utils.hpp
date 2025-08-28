#pragma once

#include <tuple>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

template<typename Floating>
using UIntOf = std::conditional_t<sizeof(Floating) == 4, uint32_t,
    std::conditional_t<sizeof(Floating) == 8, uint64_t, void>
>;

template<typename Floating>
inline std::tuple<uint32_t, uint32_t, UIntOf<Floating>>
get_components(const Floating slope) {
    using UInt = UIntOf<Floating>;

    static_assert(!std::is_void_v<UInt>, "Unsupported floating type");

    constexpr int total_bits = sizeof(Floating) * 8;
    constexpr int exp_bits   = (sizeof(Floating) == 4 ? 8 : 11);
    constexpr int mant_bits  = total_bits - exp_bits - 1;

    UInt bits{};
    std::memcpy(&bits, &slope, sizeof(Floating));

    uint32_t sign = 0;
    sign = (bits >> (total_bits - 1)) & 0x1;

    uint32_t exponent = (bits >> mant_bits) & ((1u << exp_bits) - 1);
    UInt mantissa = bits & ((UInt(1) << mant_bits) - 1);

    return {sign, exponent, mantissa};
}

template<typename T>
inline std::unordered_map<T, size_t> compute_frequencies(const std::vector<T> &data) {
    std::unordered_map<T, size_t> frequencies;
    
    if(!data.empty()) [[likely]] {
        for(const T &element : data)
            frequencies[element]++;
    }

    return frequencies;
}

template<typename T>
inline double compute_entropy(const std::vector<T> &data) {
    if(data.empty()) [[unlikely]]
        return 0.0;

    std::unordered_map<T, size_t> frequencies = compute_frequencies(data);
    
    const double n = static_cast<double>(data.size());
    double entropy = 0.0;

    for(const auto &[element, frequency] : frequencies) {
        const double ratio = static_cast<double>(frequency) / n;
        entropy -= ratio * std::log2(ratio);
    }

    return entropy;
}

template<typename T>
inline sdsl::int_vector<> build_packed_vector(std::vector<T> &vec, const T min) {
    static_assert(std::is_integral_v<T>, "Unsupported type");
    std::transform(vec.begin(), vec.end(), vec.begin(),
        [min](T v) { return v - min; });

    sdsl::int_vector<> packed_vec(vec.size());
    std::copy(vec.begin(), vec.end(), packed_vec.begin());
    sdsl::util::bit_compress(packed_vec);

    return packed_vec;
}