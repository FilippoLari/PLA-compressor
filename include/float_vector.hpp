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

template<typename Floating = float, bool AllPositive = true, 
         bool AllNegative = false>
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

    sdsl::int_vector<> mantissae;
    sdsl::int_vector<> mantissa_mapping;

    sdsl::int_vector<> exponents;
    sdsl::int_vector<> exponent_mapping;

public:
    
    float_vector() = default;

    explicit float_vector(const std::vector<Floating> &slopes) {
        if(slopes.size() == 0) [[unlikely]]
            return;
        
        if constexpr (DifferentSigns)
            signs = sdsl::bit_vector(slopes.size(), 0);

        mantissa_mapping = sdsl::int_vector<>(slopes.size());
        std::unordered_map<UInt, uint64_t> mantissa_dict;
        std::vector<UInt> tmp_mantissae;

        exponent_mapping = sdsl::int_vector<>(slopes.size());
        std::unordered_map<UInt, uint64_t> exponent_dict;
        std::vector<uint32_t> tmp_exponents;

        for(uint64_t i = 0; i < slopes.size(); ++i) {
            const auto [sign, exponent, mantissa] = get_components(slopes[i]);

            if(mantissa_dict.count(mantissa) == 0) {
                mantissa_dict[mantissa] = tmp_mantissae.size();
                tmp_mantissae.push_back(mantissa);
            }
            
            mantissa_mapping[i] = mantissa_dict[mantissa];

            if(exponent_dict.count(exponent) == 0) {
                exponent_dict[exponent] = tmp_exponents.size();
                tmp_exponents.push_back(exponent);
            }
            
            exponent_mapping[i] = exponent_dict[exponent];

            if constexpr (DifferentSigns)
                signs[i] = (sign > 0) ? 1 : 0;
        }

        sdsl::util::bit_compress(mantissa_mapping);

        mantissae = sdsl::int_vector<>(tmp_mantissae.size());
        std::copy(tmp_mantissae.begin(), tmp_mantissae.end(), mantissae.begin());
        sdsl::util::bit_compress(mantissae);

        sdsl::util::bit_compress(exponent_mapping);

        exponents = sdsl::int_vector<>(tmp_exponents.size());
        std::copy(tmp_exponents.begin(), tmp_exponents.end(), exponents.begin());
        sdsl::util::bit_compress(exponents);

        std::vector<uint32_t> tmp_m;
        std::vector<uint32_t> tmp_e;
        for(const auto &slope : slopes) {
            const auto [s, e, mantissa] = get_components(slope);
            tmp_m.push_back(mantissa);
            tmp_e.push_back(e);
        }
        huffman_vector<uint32_t> hv_m(tmp_m);
        huffman_vector<uint32_t> hv_e(tmp_e);

        std::cout << double(hv_m.size()) / double(slopes.size()) << std::endl;
        std::cout << double(hv_e.size()) / double(slopes.size()) << std::endl;

        /*std::cout << "Slopes: " << slopes.size() << std::endl;
        std::cout << "Unique mantissae: " << mantissae.size() << std::endl;
        std::cout << "Unique exponents: " << exponents.size() << std::endl;*/
    }

    inline Floating access(size_t i) const {
        return (*this)[i];
    }

    const Floating operator[](size_t i) const {
        UInt mantissa = mantissae[mantissa_mapping[i]];
        uint32_t exponent = exponents[exponent_mapping[i]];
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

    uint64_t size() const {
        return (signs.bit_size() +
                sdsl::size_in_bytes(mantissae) + sdsl::size_in_bytes(mantissa_mapping) +
                sdsl::size_in_bytes(exponents) + sdsl::size_in_bytes(exponent_mapping)) * CHAR_BIT;
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
