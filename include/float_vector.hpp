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
#include "dist_vector.hpp"
#include "pfor_vector.hpp"

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

    static_assert(AllPositive || AllNegative,
              "At least one of AllPositive or AllNegative must be true");

    static constexpr bool DifferentSigns = !AllPositive && !AllNegative;

    static constexpr int total_bits = sizeof(Floating) * 8;
    static constexpr int exp_bits = (sizeof(Floating) == 4 ? 8 : 11);
    static constexpr int mant_bits = total_bits - exp_bits - 1;

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
            const auto [sign, exponent, mantissa] = get_components(data[i]);

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
        
        UInt bits = 0;

        bits |= (mantissa & ((UInt(1) << mant_bits) - 1));
        bits |= (UInt(exponent) & ((1u << exp_bits) - 1)) << mant_bits;
        bits |= (UInt(sign & 0x1) << (total_bits - 1));

        Floating result;
        std::memcpy(&result, &bits, sizeof(Floating));
        return result;
    }

    inline uint64_t size() const {
        return signs.bit_size() + mantissae.size() + exponents.size();
    }

private:

    inline std::tuple<uint32_t, uint32_t, UInt> get_components(const Floating slope) const {
        UInt bits;
        std::memcpy(&bits, &slope, sizeof(Floating));
        uint32_t sign;

        if constexpr (DifferentSigns)
            sign = (bits >> (total_bits - 1)) & 0x1;

        uint32_t exponent = (bits >> mant_bits) & ((1u << exp_bits) - 1);
        UInt mantissa = bits & ((UInt(1) << mant_bits) - 1);

        return {sign, exponent, mantissa};
    }

};

using huff_float_vector = float_vector<huffman_vector<uint32_t, 64>, float, true, false>;

using dist_float_vector = float_vector<dist_vector<uint32_t>, float, true, false>;

using pfor_float_vector = float_vector<pfor_vector<uint32_t>, float, true, false>;
