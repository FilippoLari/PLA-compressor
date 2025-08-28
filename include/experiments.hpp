#pragma once

#include <unordered_map>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "piecewise_linear_model.hpp"
#include "slope_compressor.hpp"
#include "utils.hpp"

using entropies = std::tuple<double, double, double, double, double, double>;

using frequencies = std::pair<std::unordered_map<uint32_t, size_t>, std::unordered_map<uint32_t, size_t>>;

using components = std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>,
                                std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>;

template<typename X, typename Y, typename Floating, bool Indexing>
components compute_slope_components(const std::vector<std::conditional_t<Indexing, X, Y>>& data,
                        const uint64_t epsilon) {
    std::vector<std::pair<Floating, Floating>> slope_ranges;
    std::vector<Floating> original_slopes;
    std::vector<Floating> opt_slopes;

    auto in_fun = [data](auto i) { 
        if constexpr (Indexing)
            return std::pair<X, Y>(data[i], i); 
        else
            return std::pair<X, Y>(i, data[i]);
    };

    auto out_fun = [&](auto cs) { 
        const X x = cs.get_first_x();
        const auto [slope, beta, gamma] = cs.get_floating_point_segment(x);
        original_slopes.push_back(slope);
        slope_ranges.push_back(cs.get_slope_range());
    };

    make_segmentation_par(data.size(), epsilon, in_fun, out_fun);

    opt_slopes = slope_compressor::compress(slope_ranges);

    const size_t n = original_slopes.size();

    std::vector<uint32_t> signs(n), opt_signs(n);
    std::vector<uint32_t> exponents(n), opt_exponents(n);
    std::vector<uint32_t> mantissae(n), opt_mantissae(n);

    for(size_t i = 0; i < n; ++i) {
        const auto [sign, exponent, mantissa] = get_components(original_slopes[i]);
        const auto [opt_sign, opt_exponent, opt_mantissa] = get_components(opt_slopes[i]);
        signs[i] = sign;
        opt_signs[i] = opt_sign;
        exponents[i] = exponent;
        opt_exponents[i] = opt_exponent;
        mantissae[i] = mantissa;
        opt_mantissae[i] = opt_mantissa;
    }

    return {signs, exponents, mantissae, opt_signs, opt_exponents, opt_mantissae};
}

template<typename X, typename Y, typename Floating, bool Indexing>
entropies slope_components_entropy(const std::vector<std::conditional_t<Indexing, X, Y>>& data,
                                    const uint64_t epsilon) {
    
    const auto [signs, exponents, mantissae,
                 opt_signs, opt_exponents, opt_mantissae] = compute_slope_components<X, Y, Floating, Indexing>(data, epsilon);

    return {compute_entropy(signs), compute_entropy(exponents), compute_entropy(mantissae),
            compute_entropy(opt_signs), compute_entropy(opt_exponents), compute_entropy(opt_mantissae)};
}

template<typename X, typename Y, typename Floating, bool Indexing>
frequencies mantissae_frequencies(const std::vector<std::conditional_t<Indexing, X, Y>>& data,
                                    const uint64_t epsilon) {
    
    const auto [signs, exponents, mantissae,
                 opt_signs, opt_exponents, opt_mantissae] = compute_slope_components<X, Y, Floating, Indexing>(data, epsilon);

    return {compute_frequencies(mantissae), compute_frequencies(opt_mantissae)};
}