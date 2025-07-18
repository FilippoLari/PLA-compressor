#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "piecewise_linear_model.hpp"
#include "float_vector.hpp"

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
    uint64_t beta_shift;

    slope_container slopes;

    uint64_t segments;

public:

    SlopeCompressedPLA() = default;

    explicit SlopeCompressedPLA(const std::vector<Y> &data, const uint64_t epsilon) {
        const uint64_t n = data.size();
        
        if(n == 0) [[unlikely]] 
            return;

        const uint64_t expected_segments = n / (epsilon * epsilon);
        
        std::vector<Floating> tmp_slopes;
        std::vector<int64_t> tmp_beta;
        std::vector<uint64_t> tmp_y;
        std::vector<X> tmp_x;

        tmp_slopes.reserve(expected_segments);
        tmp_beta.reserve(expected_segments);
        tmp_y.reserve(expected_segments);
        tmp_x.reserve(expected_segments);

        int64_t min_beta = std::numeric_limits<int64_t>::max();

        auto in_fun = [data](auto i) { return std::pair<X,Y>(i, data[i]); };
        auto out_fun = [&](auto cs) { 
            const X x = cs.get_first_x();
            const auto [slope, beta, _] = cs.get_floating_point_segment(x, 0); 
            const Y y = data[x];
            const int64_t delta = beta - y;
            min_beta = (delta < min_beta) ? delta : min_beta;
            tmp_slopes.push_back(slope); 
            tmp_beta.push_back(delta);
            tmp_y.push_back(y);
            //tmp_y.push_back(tmp_y.size() ? y - tmp_y[0] : 0);
            tmp_x.push_back(x);
        };

        make_segmentation_par(n, epsilon, in_fun, out_fun);

        segments = tmp_slopes.size();

        beta_shift = min_beta;

        slopes = slope_container(tmp_slopes);

        x = sdsl::sd_vector<>(tmp_x.begin(), tmp_x.end());
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        y = sux::bits::EliasFano<>(tmp_y, tmp_y.back() + 1);

        betas = build_packed_vector(tmp_beta, min_beta);

        /*std::cout << "seg: " << segments << std::endl;
        std::cout << "slopes: " << double(slopes.size()) / double(segments)  << std::endl;
        std::cout << "x: " << double(sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)) * CHAR_BIT / double(segments) << std::endl;
        std::cout << "y: " << double(y.bitCount()) / double(segments) << std::endl;
        std::cout << "beta: " << double(sdsl::size_in_bytes(betas) * CHAR_BIT) / double(segments) << std::endl;*/
    }

    [[nodiscard]] Y predict(const X &x) {
        uint64_t i = rank_x.rank(x);
        Floating slope_i = slopes[i];
        X x_i = select_x(i + 1);
        Y y_i = y.select(i);
        uint64_t beta_i = y_i + (betas[i] - beta_shift);
        return Y(double(x - x_i) * slope_i) + beta_i;
    }

    inline size_t size() {
        return (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                + sdsl::size_in_bytes(betas)
                + sizeof(beta_shift)) * CHAR_BIT
                + y.bitCount()
                + slopes.size();
    }

    inline size_t bps() {
        return double(size()) / double(segments);
    }

private:

    /**
     * 
     */
    sdsl::int_vector<> build_packed_vector(std::vector<int64_t> &vec, int64_t &min) const {
        if(min < 0) [[likely]] {
            std::transform(vec.begin(), vec.end(), vec.begin(),
               [min](int64_t v) { return v + (min * (-1)); });
            min *= -1;
        }
        
        sdsl::int_vector<> packed_vec(vec.size());
        std::copy(vec.begin(), vec.end(), packed_vec.begin());
        sdsl::util::bit_compress(packed_vec);

        return packed_vec;
    }

};