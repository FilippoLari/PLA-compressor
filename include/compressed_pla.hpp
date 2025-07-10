#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>
#include <map>

#include <sdsl/vlc_vector.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/sd_vector.hpp>
#include <sdsl/util.hpp>

#include "sux/bits/EliasFano.hpp"

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

    sux::bits::EliasFano<> y;

    sdsl::vlc_vector<> last_y;

    sdsl::int_vector<> betas;
    sdsl::int_vector<> gammas;

    uint64_t beta_shift;
    uint64_t gamma_shift;

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

        std::vector<uint64_t> tmp_y(n_segments);
        std::vector<int64_t> tmp_last_y(n_segments);
        std::vector<int64_t> tmp_betas(n_segments);
        std::vector<int64_t> tmp_gammas(n_segments);

        int64_t min_beta = std::numeric_limits<int64_t>::max();
        int64_t min_gamma = std::numeric_limits<int64_t>::max();

        for(uint64_t i = 0; i < n_segments; ++i) {
            // first and last covered x-value by the i-th segment
            const X first_x = segments[i].get_first_x();
            const X last_x = (i < n_segments - 1) ? segments[i + 1].get_first_x() - 1 : n - 1;
            
            bv_x[first_x] = 1;

            // first covered y-value of the i-th segment
            tmp_y[i] = data[first_x];

            // last covered y-value y' of the i-th segment as
            // delta with respect to the first covered one.
            // This is because y'_i \in [y_i, y_i+1]
            tmp_last_y[i] = data[last_x] - tmp_y[i];

            auto [slope, beta, gamma] = segments[i].get_floating_point_segment(first_x, last_x);

            // intercept of the i-th segment as delta with respect
            // to the first covered y-value of the sequence
            tmp_betas[i] = beta - data[first_x];

            // same but for the last y-value given by the i-th segment
            tmp_gammas[i] = gamma - data[last_x];

            min_beta = (tmp_betas[i] < min_beta) ? tmp_betas[i] : min_beta;
            min_gamma = (tmp_gammas[i] < min_gamma) ? tmp_gammas[i] : min_gamma;
        }

        betas = build_packed_vector(tmp_betas, min_beta);
        beta_shift = min_beta;

        gammas = build_packed_vector(tmp_gammas, min_gamma);
        gamma_shift = min_gamma;

        /*for(uint64_t i = 0; i < gammas.size(); ++i) {
            std::cout << gammas[i] << std::endl;
            assert(gammas[i] <= (2*(Epsilon + 3)));
        }*/

        // initialize the ef sequences containing
        // the first covered x and last covered y value
        x = sdsl::sd_vector<>(bv_x);
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        y = sux::bits::EliasFano<>(tmp_y, tmp_y.back() + 1);

        last_y = sdsl::vlc_vector<>(tmp_last_y);

        /*for(uint64_t i = 0; i < n_segments; ++i) {
            assert(betas[i] >= 0);
            if(i == n_segments - 1)
                assert(data.back() ==  y.select(i));
            else
                assert(data[segments[i + 1].get_first_x() - 1] ==  y.select(i));
        }

        for(uint64_t i = 1; i < n_segments; ++i)
            assert(segments[i].get_first_x() == select_x(i + 1));*/
    }

    [[nodiscard]] Y predict(const X x) const {
        return 0;
    }

    /**
     * @return the size in bits of the data structure
     */
    size_t size() {
        return y.bitCount() + 
                (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                + sdsl::size_in_bytes(last_y)
                + sdsl::size_in_bytes(betas)
                + sdsl::size_in_bytes(gammas)
                + sizeof(n_segments)) * CHAR_BIT;
    }

    std::map<std::string, size_t> components_size() {
        std::map<std::string, size_t> components;
        components["first_y"] = y.bitCount();
        components["last_y"] = sdsl::size_in_bytes(last_y) * CHAR_BIT;
        components["first_x"] = (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) +
                                     sdsl::size_in_bytes(select_x)) * CHAR_BIT;
        components["betas"] = sdsl::size_in_bytes(betas) * CHAR_BIT;
        components["gammas"] = sdsl::size_in_bytes(gammas) * CHAR_BIT;
        components["other"] = (sizeof(n_segments) + sizeof(beta_shift) + 
                                sizeof(gamma_shift)) * CHAR_BIT;
        return components;
    }

    /**
     * @return the average number of bits per segment
     */
    double bps() {
        return double(size()) / double(n_segments); 
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