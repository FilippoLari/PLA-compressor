#pragma once

#include <unordered_map>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "piecewise_linear_model.hpp"
#include "slope_compressor.hpp"
#include "float_vector.hpp"
#include "utils.hpp"

#include "sdsl/sd_vector.hpp"
#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

#include "sux/bits/EliasFano.hpp"

template<typename X, typename Y, typename Floating, 
            class slope_container, bool Indexing = true>
class SlopeCompressedPLA {

    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    using pla_model = OptimalPiecewiseLinearModel<X, Y>;

    sdsl::sd_vector<> x;
    sdsl::rank_support_sd<> rank_x;
    sdsl::select_support_sd<> select_x;

    sux::bits::EliasFano<> y;

    sdsl::int_vector<> betas;
    int64_t beta_shift;

    slope_container slopes;

    uint64_t segments;
    size_t n;

    X last_x;

public:

    SlopeCompressedPLA() = default;

    explicit SlopeCompressedPLA(const std::vector<std::conditional_t<Indexing, X, Y>>& data,
                                     const uint64_t epsilon) : n(data.size()) {        
        if(n == 0) [[unlikely]] 
            return;

        const uint64_t expected_segments = n / (epsilon * epsilon);
        
        std::vector<std::pair<Floating, Floating>> slope_ranges;
        std::vector<int64_t> tmp_beta;
        std::vector<uint64_t> tmp_y;
        std::vector<X> bv_x;

        tmp_beta.reserve(expected_segments);
        tmp_y.reserve(expected_segments);
        bv_x.reserve(expected_segments);

        beta_shift = std::numeric_limits<int64_t>::max();

        std::vector<std::pair<long double, long double>> xy_pairs;

        auto in_fun = [data](auto i) { 
            if constexpr (Indexing)
                return std::pair<X, Y>(data[i], i); 
            else
                return std::pair<X, Y>(i, data[i]);
        };

        auto out_fun = [&](auto cs) { 
            const X x = cs.get_first_x();
            const Y y = cs.get_first_y();
            slope_ranges.push_back(cs.get_slope_range());
            tmp_y.push_back(y);
            bv_x.push_back(x);
            xy_pairs.push_back(cs.get_intersection());
        };

        make_segmentation_par(n, epsilon, in_fun, out_fun);

        last_x = bv_x.back();

        segments = slope_ranges.size();

        std::vector<Floating> min_entropy_slopes = slope_compressor::compress(slope_ranges);

        slopes = slope_container(min_entropy_slopes);

        x = sdsl::sd_vector<>(bv_x.begin(), bv_x.end());
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        // compute the new intercepts
        for(size_t i = 0; i < min_entropy_slopes.size(); ++i) {
            auto [i_x, i_y] = xy_pairs[i];
            Floating slope = min_entropy_slopes[i];
            int64_t beta = (int64_t) std::round(i_y - (i_x - select_x(i + 1)) * slope);
            int64_t delta = beta - tmp_y[i];
            
            assert(std::abs(delta) <= 2*(epsilon + 1));
            beta_shift = (delta < beta_shift) ? delta : beta_shift;
            tmp_beta.push_back(delta);
        }

        y = sux::bits::EliasFano<>(tmp_y, tmp_y.back() + 1);

        betas = build_packed_vector<int64_t>(tmp_beta, beta_shift);
    }

    [[nodiscard]] int64_t predict(const X &x) {
        size_t i;
        
        if(x >= last_x) [[unlikely]]
            i = segments;
        else [[likely]]
            i = rank_x(x + 1);

        Floating slope_i = slopes[i - 1];
        X x_i = select_x(i);
        Y y_i = y.select(i - 1);
        int64_t beta_i = static_cast<int64_t>(y_i) 
                            + (static_cast<int64_t>(betas[i - 1]) + beta_shift);

        if constexpr (std::is_same_v<X, int64_t> || std::is_same_v<X, int32_t>)
            return static_cast<int64_t>(slope_i * double(static_cast<std::make_unsigned_t<X>>(x) - x_i) + beta_i);
        else
            return static_cast<int64_t>(slope_i * double(x - x_i) + beta_i);
    }

    inline size_t size() {
        return (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                + sdsl::size_in_bytes(betas)
                + sizeof(beta_shift)) * CHAR_BIT
                + y.bitCount()
                + slopes.size();
    }

    inline double bps() {
        return double(size()) / double(segments);
    }

    std::unordered_map<std::string, size_t> components_size() {
        std::unordered_map<std::string, size_t> components;
        components["first_y"] = y.bitCount();
        components["first_x"] = (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) +
                                     sdsl::size_in_bytes(select_x)) * CHAR_BIT;
        components["betas"] = sdsl::size_in_bytes(betas) * CHAR_BIT;
        components["slopes"] = slopes.size();
        components["other"] = (sizeof(segments) + sizeof(beta_shift) + sizeof(n)) * CHAR_BIT;
        return components;
    }
};