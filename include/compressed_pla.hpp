#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>
#include <map>

#include "sdsl/vlc_vector.hpp"
#include "sdsl/int_vector.hpp"
#include "sdsl/sd_vector.hpp"
#include "sdsl/util.hpp"

#include "sux/bits/EliasFano.hpp"

#include "piecewise_linear_model.hpp"

template<typename X, typename Y>
class CompressedPLA {

    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    using pla_model = OptimalPiecewiseLinearModel<X, Y>;
    using segment = OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment;

    sdsl::sd_vector<> x;
    sdsl::rank_support_sd<> rank_x;
    sdsl::select_support_sd<> select_x;

    sux::bits::EliasFano<> y;
    //sdsl::sd_vector<> y;
    //sdsl::select_support_sd<> select_y;

    sux::bits::EliasFano<> last_y;
    //sdsl::sd_vector<> last_y;
    //sdsl::select_support_sd<> select_last_y;

    //sdsl::vlc_vector<> last_y;

    sdsl::int_vector<> betas;
    sdsl::int_vector<> gammas;

    uint64_t beta_shift;
    uint64_t gamma_shift;

    uint64_t segments;

public:

    CompressedPLA() = default;

    explicit CompressedPLA(const std::vector<Y> &data, const uint64_t epsilon) {
        if(data.size() == 0) [[unlikely]]
            return;

        const uint64_t n = data.size();
        const Y u = data.back(); // the sequence is increasing

        std::vector<segment> segments_v;
        segments_v.reserve(n / (epsilon * epsilon));

        auto in_fun = [data](auto i) { return std::pair<X,Y>(i, data[i]); };
        auto out_fun = [&segments_v](auto cs) { segments_v.emplace_back(cs); };

        segments = make_segmentation_par(n, epsilon, in_fun, out_fun);

        sdsl::bit_vector bv_x(n + 1, 0);

        betas = sdsl::int_vector<>(segments);
        gammas = sdsl::int_vector<>(segments);

        std::vector<uint64_t> tmp_y(segments);
        std::vector<uint64_t> tmp_last_y(segments);
        std::vector<int64_t> tmp_betas(segments);
        std::vector<int64_t> tmp_gammas(segments);

        beta_shift= std::numeric_limits<int64_t>::max();
        gamma_shift = std::numeric_limits<int64_t>::max();

        for(uint64_t i = 0; i < segments; ++i) {
            // first and last covered x-value by the i-th segment
            const X first_x = segments_v[i].get_first_x();
            const X last_x = (i < segments - 1) ? segments_v[i + 1].get_first_x() - 1 : n - 1;
            
            bv_x[first_x] = 1;

            // first covered y-value of the i-th segment
            tmp_y[i] = data[first_x];

            // last covered y-value of the i-th segment
            tmp_last_y[i] = data[last_x];

            auto [slope, beta, gamma] = segments_v[i].get_floating_point_segment(first_x, last_x);

            // intercept of the i-th segment as delta with respect
            // to the first covered y-value of the sequence
            tmp_betas[i] = beta - data[first_x];

            // same but for the last y-value given by the i-th segment
            tmp_gammas[i] = gamma - data[last_x];

            beta_shift = (tmp_betas[i] < beta_shift) ? tmp_betas[i] : beta_shift;
            gamma_shift = (tmp_gammas[i] < gamma_shift) ? tmp_gammas[i] : gamma_shift;
        }

        betas = build_packed_vector(tmp_betas, beta_shift);

        gammas = build_packed_vector(tmp_gammas, gamma_shift);

        // initialize the ef sequences containing
        // the first covered x and last covered y value
        x = sdsl::sd_vector<>(bv_x);
        sdsl::util::init_support(rank_x, &x);
		sdsl::util::init_support(select_x, &x);

        y = sux::bits::EliasFano<>(tmp_y, tmp_y.back() + 1);

        last_y = sux::bits::EliasFano<>(tmp_last_y, tmp_last_y.back() + 1);
    }

    /**
     * Given an x-value retrieve the segment covering such abscissa
     * and return the corresponding y-value given by the segment.
     * 
     * todo: add a fake segment at construction time to avoid corner cases
     * 
     * @param x a value lying on the x-axis
     * @return the y-value given by the segment covering x
     */
    [[nodiscard]] Y predict(const X x) {
        const uint64_t i = rank_x(x + 1);
        X x_i = select_x(i);
        X x_ii = select_x(i + 1);
        Y y_i = y.select(i - 1);
        Y y_ii = last_y.select(i);

        int64_t beta_i = static_cast<int64_t>(y_i) 
                            - (static_cast<int64_t>(betas[i - 1]) + beta_shift);
        
        int64_t gamma_i = static_cast<int64_t>(y_ii) 
                            - (static_cast<int64_t>(gammas[i - 1]) + gamma_shift);
        
        double slope_i = static_cast<double>(gamma_i - beta_i) / static_cast<double>(x_ii - x_i);

        if constexpr (std::is_same_v<X, int64_t> || std::is_same_v<X, int32_t>)
            return static_cast<uint64_t>(static_cast<double>(std::make_unsigned<X>(x) - x_i) * slope_i) + beta_i;
        else
            return static_cast<uint64_t>(static_cast<double>(x - x_i) * slope_i) + beta_i;
    }

    /**
     * @return the size in bits of the data structure
     */
    size_t size() {
        return y.bitCount() + last_y.bitCount() + 
                (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) + sdsl::size_in_bytes(select_x)
                //+ sdsl::size_in_bytes(y) + sdsl::size_in_bytes(select_y)
                //+ sdsl::size_in_bytes(last_y) + sdsl::size_in_bytes(select_last_y)
                + sdsl::size_in_bytes(betas)
                + sdsl::size_in_bytes(gammas)
                + sizeof(segments)) * CHAR_BIT;
    }

    /**
     * @return a map containing the size of each component in bits
     */
    std::map<std::string, size_t> components_size() {
        std::map<std::string, size_t> components;
        components["first_y"] = y.bitCount();
        //components["first_y"] = (sdsl::size_in_bytes(y) + sdsl::size_in_bytes(select_y)) * CHAR_BIT;
        //components["last_y"] = (sdsl::size_in_bytes(last_y) + sdsl::size_in_bytes(select_last_y)) * CHAR_BIT;
        components["last_y"] = last_y.bitCount();
        components["first_x"] = (sdsl::size_in_bytes(x) + sdsl::size_in_bytes(rank_x) +
                                     sdsl::size_in_bytes(select_x)) * CHAR_BIT;
        components["betas"] = sdsl::size_in_bytes(betas) * CHAR_BIT;
        components["gammas"] = sdsl::size_in_bytes(gammas) * CHAR_BIT;
        components["other"] = (sizeof(segments) + sizeof(beta_shift) + 
                                sizeof(gamma_shift)) * CHAR_BIT;
        return components;
    }

    /**
     * @return the average number of bits per segment
     */
    double bps() {
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