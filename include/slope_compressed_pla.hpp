#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "piecewise_linear_model.hpp"
#include "slope_compressor.hpp"
#include "float_vector.hpp"
#include "rle_vector.hpp"

#include "sdsl/sd_vector.hpp"
#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

#include "sux/bits/EliasFano.hpp"

template<typename X, typename Y, typename Floating, class slope_container>
class SlopeCompressedPLA {

    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    using pla_model = OptimalPiecewiseLinearModel<X, Y>;

    sdsl::sd_vector<> x;
    sdsl::rank_support_sd<> rank_x;
    sdsl::select_support_sd<> select_x;

    sux::bits::EliasFano<> y;

    sdsl::int_vector<> betas;
    //rle_vector<Y> betas;
    int64_t beta_shift;

    slope_container slopes;

    uint64_t segments;

public:

    SlopeCompressedPLA() = default;

    explicit SlopeCompressedPLA(const std::vector<Y> &data, const uint64_t epsilon) {
        const uint64_t n = data.size();
        
        if(n == 0) [[unlikely]] 
            return;

        const uint64_t expected_segments = n / (epsilon * epsilon);
        
        std::vector<std::pair<Floating, Floating>> slope_ranges;
        //std::vector<Floating> tmp_slopes;
        std::vector<int64_t> tmp_beta;
        std::vector<uint64_t> tmp_y; // todo: fix uint64_t

        // n + 1 positions allows us to avoid an if at query time
        sdsl::bit_vector bv_x(n + 1, 0);

        //tmp_slopes.reserve(expected_segments);
        tmp_beta.reserve(expected_segments);
        tmp_y.reserve(expected_segments);

        beta_shift = std::numeric_limits<int64_t>::max();

        std::vector<std::pair<long double, long double>> xy_pairs;

        auto in_fun = [data](auto i) { return std::pair<X,Y>(i, data[i]); };
        auto out_fun = [&](auto cs) { 
            const X x = cs.get_first_x();
            //const auto [slope, beta, _] = cs.get_floating_point_segment(x, 0); 
            const Y y = data[x];
            //const int64_t delta = int64_t(y) - int64_t(beta);
            //beta_shift = (delta < beta_shift) ? delta : beta_shift;
            slope_ranges.push_back(cs.get_slope_range());
            //tmp_slopes.push_back(slope); 
            //tmp_beta.push_back(delta);
            tmp_y.push_back(y);
            bv_x[x] = 1;

            xy_pairs.push_back(cs.get_intersection());
        };

        make_segmentation_par(n, epsilon, in_fun, out_fun);

        //segments = tmp_slopes.size();
        segments = slope_ranges.size();

        std::vector<Floating> min_entropy_slopes = slope_compressor::compress(slope_ranges);

        slopes = slope_container(min_entropy_slopes);

        x = sdsl::sd_vector<>(bv_x);
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        // compute the new intercepts
        for(size_t i = 0; i < min_entropy_slopes.size(); ++i) {
            auto [i_x, i_y] = xy_pairs[i];
            Floating slope = min_entropy_slopes[i];
            int64_t beta = (int64_t) std::round(i_y - (i_x - select_x(i + 1)) * slope);
            int64_t delta = int64_t(tmp_y[i]) - int64_t(beta);
            assert(std::abs(delta) <= 2*(epsilon + 2));
            beta_shift = (delta < beta_shift) ? delta : beta_shift;
            tmp_beta.push_back(delta);
        }

        y = sux::bits::EliasFano<>(tmp_y, tmp_y.back() + 1);

        betas = build_packed_vector(tmp_beta, beta_shift);
        //build_packed_vector(tmp_beta, beta_shift);

        //betas = rle_vector<Y>(tmp_beta);
    }

    [[nodiscard]] Y predict(const X &x) {
        const uint64_t i = rank_x(x + 1);
        Floating slope_i = slopes[i - 1];
        X x_i = select_x(i);
        Y y_i = y.select(i - 1);
        int64_t beta_i = static_cast<int64_t>(y_i) 
                            - (static_cast<int64_t>(betas[i - 1]) + beta_shift);

        if constexpr (std::is_same_v<X, int64_t> || std::is_same_v<X, int32_t>)
            return static_cast<uint64_t>(static_cast<double>(std::make_unsigned<X>(x) - x_i) * slope_i) + beta_i;
        else
            return static_cast<uint64_t>(static_cast<double>(x - x_i) * slope_i) + beta_i;
    }

    inline size_t size() {
        return (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                //+ sdsl::size_in_bytes(betas)
                + sizeof(beta_shift)) * CHAR_BIT
                + betas.size()
                + y.bitCount()
                + slopes.size();
    }

    inline double bps() {
        return double(size()) / double(segments);
    }

private:

    sdsl::int_vector<> build_packed_vector(std::vector<int64_t> &vec, int64_t min) const {
        std::transform(vec.begin(), vec.end(), vec.begin(),
            [min](int64_t v) { return v - min; });

        sdsl::int_vector<> packed_vec(vec.size());
        std::copy(vec.begin(), vec.end(), packed_vec.begin());
        sdsl::util::bit_compress(packed_vec);

        return packed_vec;
    }

};