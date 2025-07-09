#pragma once

#include <climits>
#include <cstddef>
#include <vector>

#include <sdsl/vlc_vector.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/sd_vector.hpp>
#include <sdsl/util.hpp>

#include "piecewise_linear_model.hpp"

template<typename X, typename Y, size_t Epsilon>
class CompressedPLA {

    static_assert(Epsilon > 0);
    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    using pla_model = OptimalPiecewiseLinearModel<X, Y>;
    using segment = OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment;

    sdsl::sd_vector<> x;
    sdsl::rank_support_sd<> rank_x;
    sdsl::select_support_sd<> select_x;

    sdsl::sd_vector<> y;
    sdsl::select_support_sd<> select_y;

    sdsl::vlc_vector<> last_y;

    sdsl::int_vector<> betas;
    sdsl::int_vector<> gammas;

    size_t n_segments;

public:

    CompressedPLA() = default;

    explicit CompressedPLA(const std::vector<Y> data) {
        if(data.size() == 0) [[unlikely]]
            return;

        const uint64_t n = data.size();
        const Y u = *std::max_element(data.begin(), data.end());

        std::vector<segment> segments;
        segments.reserve(n / (Epsilon * Epsilon));

        auto in_fun = [data](auto i) { return std::pair<X,Y>(i, data[i]); };
        auto out_fun = [&segments](auto cs) { segments.emplace_back(cs); };

        n_segments = make_segmentation_par(n, Epsilon, in_fun, out_fun);

        sdsl::bit_vector bv_x(n, 0);
        sdsl::bit_vector bv_y(u + 1, 0);

        betas = sdsl::int_vector<>(n_segments);
        gammas = sdsl::int_vector<>(n_segments);

        for(uint64_t i = 0; i < n_segments; ++i) {
            const X x_i = segments[i].get_first_x();
            bv_x[x_i] = 1;

            if(i > 0) [[likely]] {
                const Y y_i = data[x_i - 1];
                bv_y[y_i] = 1;
            }

            auto [slope, beta] = segments[i].get_floating_point_segment(x_i);
            betas[i] = beta;
        }

        // last covered y is always the last value of the sequence
        bv_y[data.back()] = 1;

        // initialize the ef sequences containing
        // the first covered x and last covered y value
        x = sdsl::sd_vector<>(bv_x);
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        y = sdsl::sd_vector<>(bv_y);
        sdsl::util::init_support(select_y, &y);

        /*for(uint64_t i = 1; i < n_segments; ++i) {
            assert(segments[i].get_first_x() == select_x(i + 1));
            std::cout << "y_i: " << data[segments[i].get_first_x() - 1] << " ef: " << select_y(i) << std::endl;
            assert(data[segments[i].get_first_x() - 1] == select_y(i));
        }*/
    }

    [[nodiscard]] Y predict(const X x) const {
        return 0;
    }

    /**
     * @return the size in bits of the data structure
     */
    size_t size() const {
        return (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                + sdsl::size_in_bytes(y) + sdsl::size_in_bytes(select_y)
                + sdsl::size_in_bytes(last_y)
                + sdsl::size_in_bytes(betas)
                + sdsl::size_in_bytes(gammas)
                + sizeof(n_segments)) * CHAR_BIT;
    }

    /**
     * @return the average number of bits per segment
     */
    double bps() const {
        return double(size()) / double(n_segments); 
    }
};