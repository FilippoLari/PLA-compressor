#pragma once

#include <unordered_map>
#include <type_traits>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <tuple>
#include <cmath>

#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

template<typename Floating>
constexpr int total_bits = sizeof(Floating) * 8;

template<typename Floating>
constexpr int exp_bits   = (sizeof(Floating) == 4 ? 8 : 11);

template<typename Floating>
constexpr int mant_bits  = total_bits<Floating> - exp_bits<Floating> - 1;

template<typename Floating>
using UIntOf = std::conditional_t<sizeof(Floating) == 4, uint32_t,
    std::conditional_t<sizeof(Floating) == 8, uint64_t, void>
>;

template<typename Floating>
inline std::tuple<uint32_t, uint32_t, UIntOf<Floating>>
get_components(const Floating slope) {
    using UInt = UIntOf<Floating>;

    static_assert(!std::is_void_v<UInt>, "Unsupported floating type");

    UInt bits{};
    std::memcpy(&bits, &slope, sizeof(Floating));

    uint32_t sign = 0;
    sign = (bits >> (total_bits<Floating> - 1)) & 0x1;

    uint32_t exponent = (bits >> mant_bits<Floating>) & ((1u << exp_bits<Floating>) - 1);
    UInt mantissa = bits & ((UInt(1) << mant_bits<Floating>) - 1);

    return {sign, exponent, mantissa};
}

template<typename Floating>
inline Floating build_float(const uint32_t sign, const uint32_t exponent,
                             const UIntOf<Floating> mantissa) {
    using UInt = UIntOf<Floating>;
    
    UInt bits = 0;

    bits |= (mantissa & ((UInt(1) << mant_bits<Floating>) - 1));
    bits |= (UInt(exponent) & ((1u << exp_bits<Floating>) - 1)) << mant_bits<Floating>;
    bits |= (UInt(sign & 0x1) << (total_bits<Floating> - 1));

    Floating result;
    std::memcpy(&result, &bits, sizeof(Floating));
    return result;
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