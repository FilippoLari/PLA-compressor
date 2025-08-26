#pragma once

#include <type_traits>
#include <cassert>
#include <cstring>
#include <optional>
#include <cstdint>
#include <climits>
#include <vector>
#include <tuple>
#include <bit>
#include <iostream>

#include "piecewise_linear_model.hpp"

class slope_compressor {

    struct range {
        uint32_t e_a, m_a;
        uint32_t e_b, m_b;
        uint64_t idx;

        // true if the original range covers negative values
        bool norm_r;

        // this is used to handle negative ranges
        // of the form [-a, b]. They are converted to [0, max(a,b)].
        // if a > b, then we need to remember that the values after b
        // were originally negative.
        uint32_t r_m;

        range(uint32_t e_a, uint32_t m_a, 
                uint32_t e_b, uint32_t m_b,
                 uint64_t idx, bool norm_r,
                  uint32_t r_m) : e_a(e_a), m_a(m_a),
                                 e_b(e_b), m_b(m_b),
                                 idx(idx), norm_r(norm_r),
                                 r_m(r_m) {}

        range(uint32_t e_a, uint32_t m_a, 
                uint32_t e_b, uint32_t m_b,
                 uint64_t idx) : e_a(e_a), m_a(m_a),
                                 e_b(e_b), m_b(m_b),
                                 idx(idx), norm_r(false), 
                                 r_m(0) {}

        bool operator<(const range &r) const { return m_a < r.m_a; }
    };

    struct merged_range {
        uint32_t m_a;
        uint32_t m_b;

        std::vector<range> ranges;

        merged_range() = default;

        merged_range(merged_range&& other) noexcept = default; 
        
        merged_range& operator=(merged_range&& other) noexcept = default;
        
        merged_range(range &&r) : m_a(r.m_a), m_b(r.m_b) {
            ranges.push_back(std::move(r));
        }

        inline void update(const uint32_t new_m_a, const uint32_t new_m_b, range &&r) {
            m_a = new_m_a;
            m_b = new_m_b;
            ranges.push_back(std::move(r));
        }

        bool operator<(const merged_range& mr) const {
            if (ranges.size() != mr.ranges.size())
                return ranges.size() > mr.ranges.size();
            else
                return m_a < mr.m_a;
        }
    };

public:
    
    static constexpr int total_bits = sizeof(float) * 8;
    static constexpr int exp_bits = (sizeof(float) == 4 ? 8 : 11);
    static constexpr int mant_bits = total_bits - exp_bits - 1;

    static std::vector<float> compress(const std::vector<std::pair<float, float>> &slope_ranges) {

        std::vector<float> slopes(slope_ranges.size());

        std::vector<range> jolly_ranges;
        std::vector<range> eq_exp_ranges;
        std::vector<range> to_merge;

        for(size_t i = 0; i < slope_ranges.size(); ++i) {
            auto& [orig_a, orig_b] = slope_ranges[i];
            
            auto [a, b, norm, remap] = normalize_range(orig_a, orig_b);

            if(orig_a < 0 || orig_b < 0) { // sanity check
                assert(a <= b);
            }

            auto [exp_a, mant_a] = get_components(a);
            auto [exp_b, mant_b] = get_components(b);

            range r(exp_a, mant_a, exp_b, mant_b, i, norm, remap);

            if(exp_a == exp_b) {
                eq_exp_ranges.push_back(r);
                assert(mant_a <= mant_b);
            } else if (exp_b > exp_a + 1 || mant_b >= mant_a)
                jolly_ranges.push_back(r);
            else
                to_merge.push_back(r);
        }

        // sort ranges by starting value of the mantissa for sweep-line merging
        std::sort(eq_exp_ranges.begin(), eq_exp_ranges.end());

        std::vector<merged_range> eq_exp_merged;

        merged_range curr_intersect(std::move(eq_exp_ranges[0]));

        for(size_t i = 1; i < eq_exp_ranges.size(); ++i) {
            const uint32_t range_min = eq_exp_ranges[i].m_a;
            const uint32_t range_max = eq_exp_ranges[i].m_b;
            const uint32_t range_exp = eq_exp_ranges[i].e_a; // notice exp_a = exp_b
            const size_t range_idx = eq_exp_ranges[i].idx;

            if(range_min > curr_intersect.m_b) {
                eq_exp_merged.push_back(std::move(curr_intersect));
                curr_intersect = merged_range(std::move(eq_exp_ranges[i]));
            } 
            else {
                curr_intersect.update(std::max(curr_intersect.m_a, range_min),
                                        std::min(curr_intersect.m_b, range_max),
                                        std::move(eq_exp_ranges[i]));
            }
        }

        eq_exp_merged.push_back(std::move(curr_intersect));

        // iteratively assign the non-jolly ranges to the
        // intersection covering the most ranges.
        // use a sorted array because updates should be infrequent
        std::sort(eq_exp_merged.begin(), eq_exp_merged.end());

        auto update_range = [&](auto it, uint32_t new_m_a, uint32_t new_m_b, uint32_t exp, size_t idx) {
            merged_range updated = std::move(*it);
            eq_exp_merged.erase(it);

            updated.update(new_m_a, new_m_b, {exp, new_m_a, exp, new_m_b, idx}); // we always use the same exponent

            auto pos = std::lower_bound(eq_exp_merged.begin(), eq_exp_merged.end(), updated);

            eq_exp_merged.insert(pos, std::move(updated));
        };

        for(range &range_to_merge : to_merge) {
            bool merged = false;

            for(auto it = eq_exp_merged.begin(); it != eq_exp_merged.end(); ++it) {
                const merged_range& curr_intersection = *it;

                // try with [a, 2^23-1] and [0, b]
                if(auto first_intersect = intersect(
                        range_to_merge.m_a, (uint32_t(1) << 23)-1,
                        curr_intersection.m_a, curr_intersection.m_b)) {

                    auto [a, b] = *first_intersect;
                    update_range(it, a, b, range_to_merge.e_a, range_to_merge.idx);
                    merged = true;
                    break;
                } else if(auto second_intersect = intersect(
                        0, range_to_merge.m_b,
                        curr_intersection.m_a, curr_intersection.m_b)) {

                    auto [a, b] = *second_intersect;
                    update_range(it, a, b, range_to_merge.e_b, range_to_merge.idx);
                    merged = true;
                    break;
                }
            }

            if(!merged) [[unlikely]] {
                eq_exp_merged.push_back(std::move(range_to_merge));
            }
        }

        // finalize the assignment
        for(const merged_range &intersection : eq_exp_merged) {
            // use the smallest mantissa
            const uint32_t mant = intersection.m_a;
            
            // notice that after the above processing
            // e_a is always the correct exponent
            for(const range &r : intersection.ranges) {
                const uint32_t sign = (r.norm_r && (mant >= r.r_m)) ? 1 : 0;
                slopes[r.idx] = build_float(sign, r.e_a, mant);
            }
        }

        // assign the mantissae of jolly ranges to the most frequent
        // mantissa of the merged ranges with equal exponent

        uint32_t most_freq_mant = eq_exp_merged.begin()->m_a;

        for(const range& r : jolly_ranges) {
            uint32_t exponent;

            if (r.e_b > r.e_a + 1) {
                exponent = r.e_a + 1;
            } else if (most_freq_mant <= r.m_b) { 
                exponent = r.e_b;
            } else {
                exponent = r.e_a;
            }

            slopes[r.idx] = build_float(0, exponent, most_freq_mant);
        }

        // sanity check
        for(size_t i = 0; i < slopes.size(); ++i) {
            if(slopes[i] < slope_ranges[i].first || slopes[i] > slope_ranges[i].second) std::cout << slope_ranges[i].first << " " << slope_ranges[i].second << " " << slopes[i] << std::endl;
            assert(slopes[i] >= slope_ranges[i].first && slopes[i] <= slope_ranges[i].second);
        }

        return slopes;
    }

    static std::optional<std::pair<uint32_t, uint32_t>>
    intersect(uint32_t a1, uint32_t b1, uint32_t a2, uint32_t b2) {
        uint32_t start = std::max(a1, a2);
        uint32_t end = std::min(b1, b2);

        if (start <= end) {
            return std::make_pair(start, end);
        }
        return std::nullopt;
    }

    static inline std::tuple<float, float, bool, uint32_t> 
    normalize_range(float a, float b) {
        uint32_t remap = 0;
        
        if (a >= 0) [[likely]]
            return {a, b, false, remap};

        if (b >= 0) {
            a = -a;
            remap = b + 1; 
            b = std::max(a, b);
            a = 0;
            return {0, b, true, remap};
        } else {
            return {-b, -a, true, remap};
        }
    }

    static inline std::pair<uint32_t, uint32_t>
    get_components(const float slope) {
        uint32_t bits;
        std::memcpy(&bits, &slope, sizeof(float));

        uint32_t exponent = (bits >> mant_bits) & ((1u << exp_bits) - 1);
        uint32_t mantissa = bits & ((uint32_t(1) << mant_bits) - 1);

        return {exponent, mantissa};
    }

    static inline float build_float(const uint32_t sign, const uint32_t exponent, const uint32_t mantissa) {
        uint32_t bits = 0;

        bits |= (mantissa & ((uint32_t(1) << mant_bits) - 1));
        bits |= (uint32_t(exponent) & ((1u << exp_bits) - 1)) << mant_bits;
        bits |= (uint32_t(sign & 0x1) << (total_bits - 1));

        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }

};
