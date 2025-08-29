#pragma once

#include <unordered_map>
#include <type_traits>
#include <cstdint>
#include <climits>
#include <vector>
#include <tuple>
#include <bit>

#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

#include "huffman_vector.hpp"
#include "pfor_vector.hpp"
#include "utils.hpp"

/**
 * A storage scheme for a sequence of floating-point values storing
 * each sign, exponent and mantissa separately.
 */
template<class compressed_seq, typename Floating = float,
         bool AllPositive = true, bool AllNegative = false>
class float_vector
{
    static_assert(std::is_floating_point_v<Floating>,
                  "Template parameter must be a floating-point type");

    using UInt = std::conditional_t<
        sizeof(Floating) == 4, uint32_t,
        std::conditional_t<sizeof(Floating) == 8, uint64_t,
                           void>
    >;
    static_assert(!std::is_void_v<UInt>,
                  "Unsupported floating-point size: only 32- and 64-bit are supported");

    static_assert(!(AllPositive && AllNegative),
                  "AllPositive and AllNegative cannot both be true");

    static constexpr bool DifferentSigns = !AllPositive && !AllNegative;

    sdsl::bit_vector signs;

    compressed_seq mantissae;

    compressed_seq exponents;

public:
    
    float_vector() = default;

    explicit float_vector(const std::vector<Floating> &data) {
        if(data.size() == 0) [[unlikely]]
            return;
        
        if constexpr (DifferentSigns) 
            signs = sdsl::bit_vector(data.size(), 0);

        std::vector<UInt> tmp_mantissae(data.size());
        std::vector<uint32_t> tmp_exponents(data.size());

        for(uint64_t i = 0; i < data.size(); ++i) {
            const auto [sign, exponent, mantissa] = get_components<Floating>(data[i]);

            tmp_mantissae[i] = mantissa;
            tmp_exponents[i] = exponent;

            if constexpr (DifferentSigns)
                signs[i] = (sign > 0) ? 1 : 0;
        }

        mantissae = compressed_seq(tmp_mantissae);
        exponents = compressed_seq(tmp_exponents);
    }

    const Floating operator[](size_t i) const {
        UInt mantissa = mantissae[i];
        uint32_t exponent = exponents[i];
        constexpr uint8_t default_sign = AllNegative ? 1 : 0;
        uint8_t sign = (DifferentSigns) ? signs[i] : default_sign;
        
        return build_float<Floating>(sign, exponent, mantissa);
    }

    inline uint64_t size() const {
        return signs.bit_size() + mantissae.size() + exponents.size();
    }

};

using huff_float_vector = float_vector<huffman_vector<uint32_t, 64>, float, true, false>;

using pfor_float_vector = float_vector<pfor_vector<uint32_t>, float, true, false>;

using pf_mixed_float_vector = float_vector<pfor_vector<uint32_t>, float, false, false>;

using huff_mixed_float_vector = float_vector<huffman_vector<uint32_t, 64>, float, false, false>;
