#pragma once

#include <unordered_map>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "sdsl/int_vector.hpp"
#include "sdsl/sd_vector.hpp"
#include "sdsl/util.hpp"

#include "sux/bits/EliasFano.hpp"

#include "piecewise_linear_model.hpp"
#include "utils.hpp"

/**
 * A compressed storage scheme for PLAs that achieves succinct space
 * under mild assumptions on the number of segments, while still supporting
 * fast query times.
 *
 * See the accompanying paper for details.
 * 
 * Complexity:
 * - Space: 2l * (log(u/l) + log(n/l) + log(2epsilon + 1) + o(1)) bits
 * - Time: O(log(l))  
 */
template<typename X, typename Y, bool Indexing = true>
class SuccinctPLA {

    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    using pla_model = OptimalPiecewiseLinearModel<X, Y>;

    using segment = typename pla_model::CanonicalSegment;

    // both indexing and compression require to store the first x and y
    // values with rank support on the x-values and select support on the y-values
    sdsl::sd_vector<> x;
    sdsl::rank_support_sd<> rank_x;
    sdsl::select_support_sd<> select_x;

    sux::bits::EliasFano<> y;

    // depending on the scenario this is interpreated as either
    // the last y-value (compression) or the last x-value (indexing)
    sux::bits::EliasFano<> last_v;

    sdsl::int_vector<> betas;
    sdsl::int_vector<> gammas;

    int64_t beta_shift;
    int64_t gamma_shift;

    uint64_t segments;
    size_t n;

    X last_x;

public:

    SuccinctPLA() = default;

    explicit SuccinctPLA(const std::vector<std::conditional_t<Indexing, X, Y>>& data,
                             const uint64_t epsilon) : n(data.size()) {
        if(n == 0) [[unlikely]]
            return;

        std::vector<segment> segments_v;
        segments_v.reserve(n / (epsilon * epsilon));

        auto in_fun = [data](auto i) { 
            if constexpr (Indexing)
                return std::pair<X, Y>(data[i], i); 
            else
                return std::pair<X, Y>(i, data[i]);
        };

        auto out_fun = [&segments_v](auto cs) { segments_v.emplace_back(cs); };

        segments = make_segmentation_par(n, epsilon, in_fun, out_fun);

        std::vector<X> bv_x;

        betas = sdsl::int_vector<>(segments);
        gammas = sdsl::int_vector<>(segments);

        std::vector<uint64_t> tmp_y(segments);
        std::vector<uint64_t> tmp_last_v(segments);
        std::vector<int64_t> tmp_betas(segments);
        std::vector<int64_t> tmp_gammas(segments);

        beta_shift = std::numeric_limits<int64_t>::max();
        gamma_shift = std::numeric_limits<int64_t>::max();

        for(size_t i = 0; i < segments; ++i) {

            const uint64_t first_x = segments_v[i].get_first_x();

            bv_x.push_back(first_x);

            const uint64_t last_y = segments_v[i].get_last_y();

            // first covered y-value of the i-th segment
            tmp_y[i] = segments_v[i].get_first_y();

            if constexpr (Indexing) 
                tmp_last_v[i] = segments_v[i].get_last_x();
            else
                tmp_last_v[i] = last_y;

            auto [slope, beta, gamma] = segments_v[i].get_floating_point_segment(first_x);

            // intercept of the i-th segment as delta with respect
            // to the first covered y-value of the sequence
            tmp_betas[i] = beta - tmp_y[i];

            // same but for the last y-value given by the i-th segment
            tmp_gammas[i] = gamma - last_y;

            beta_shift = (tmp_betas[i] < beta_shift) ? tmp_betas[i] : beta_shift;
            gamma_shift = (tmp_gammas[i] < gamma_shift) ? tmp_gammas[i] : gamma_shift;
        }

        betas = build_packed_vector<int64_t>(tmp_betas, beta_shift);

        gammas = build_packed_vector<int64_t>(tmp_gammas, gamma_shift);

        last_x = bv_x.back();
        
        if constexpr (Indexing)
            tmp_y.push_back(n);
        else
            bv_x.push_back(n);

        // initialize the Elias-Fano sequences 
        x = sdsl::sd_vector<>(bv_x.begin(), bv_x.end());
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        y = sux::bits::EliasFano<>(tmp_y, tmp_y.back() + 1);

        last_v = sux::bits::EliasFano<>(tmp_last_v, tmp_last_v.back() + 1);
    }

    /**
     * Given an x-value retrieve the segment covering such abscissa
     * and return the corresponding y-value given by the same segment.
     * 
     * @param x a value lying on the x-axis
     * @return the y-value given by the segment covering x
     */
    [[nodiscard]] int64_t predict(const X &x) {
        size_t i;
        
        if(x >= last_x) [[unlikely]]
            i = segments;
        else [[likely]]
            i = rank_x(x + 1);

        X x_i = select_x(i), x_ii;
        Y y_i = y.select(i - 1), y_ii;

        if constexpr (Indexing) {
            x_ii = last_v.select(i - 1);
            y_ii = y.select(i) - 1;
        } else {
            x_ii = select_x(i + 1) - 1;
            y_ii = last_v.select(i - 1);
        }

        int64_t beta_i = static_cast<int64_t>(y_i) 
                            + (static_cast<int64_t>(betas[i - 1]) + beta_shift);
        
        int64_t gamma_i = static_cast<int64_t>(y_ii) 
                            + (static_cast<int64_t>(gammas[i - 1]) + gamma_shift);
        
        double slope_i;
        
        if(x_ii > x_i) [[likely]]
            slope_i = static_cast<double>(gamma_i - beta_i) / static_cast<double>(x_ii - x_i);
        else [[unlikely]]
            slope_i = 0;

        if constexpr (std::is_same_v<X, int64_t> || std::is_same_v<X, int32_t>)
            return static_cast<int64_t>(slope_i * double(static_cast<std::make_unsigned_t<X>>(x) - x_i) + beta_i);
        else
            return static_cast<int64_t>(slope_i * double(x - x_i) + beta_i);
    }

    /**
     * @return the size in bits of the data structure
     */
    size_t size() {
        return y.bitCount() + last_v.bitCount() + 
                (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                + sdsl::size_in_bytes(betas)
                + sdsl::size_in_bytes(gammas)
                + sizeof(segments)) * CHAR_BIT;
    }

    /**
     * @return a map containing the size of each component in bits
     */
    std::unordered_map<std::string, size_t> components_size() {
        std::unordered_map<std::string, size_t> components;
        components["first_y"] = y.bitCount();
        components["last_v"] = last_v.bitCount();
        components["first_x"] = (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) +
                                     sdsl::size_in_bytes(select_x)) * CHAR_BIT;
        components["betas"] = sdsl::size_in_bytes(betas) * CHAR_BIT;
        components["gammas"] = sdsl::size_in_bytes(gammas) * CHAR_BIT;
        components["other"] = (sizeof(segments) + sizeof(beta_shift) + sizeof(n) +
                                sizeof(gamma_shift)) * CHAR_BIT;
        return components;
    }

    /**
     * @return the average number of bits per segment
     */
    double bps() {
        return double(size()) / double(segments); 
    }
};