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

        range(uint32_t e_a, uint32_t m_a, 
                uint32_t e_b, uint32_t m_b,
                 uint64_t idx) : e_a(e_a), m_a(m_a),
                                 e_b(e_b), m_b(m_b),
                                 idx(idx) {}

        bool operator<(const range &r) const { return m_a < r.m_a; }
    };

    struct merged_range {
        uint32_t m_a;
        uint32_t m_b;

        // keep track of the original slope indexes and
        // the original exponenets of the intersected ranges
        std::vector<size_t> indexes;
        std::vector<uint32_t> exponents;

        merged_range() = default;

        merged_range(merged_range&& other) noexcept = default; 
        
        merged_range& operator=(merged_range&& other) noexcept = default;
        
        merged_range(uint32_t m_a, uint32_t m_b, uint32_t exp, size_t i) : m_a(m_a), m_b(m_b) {
            exponents.push_back(exp);
            indexes.push_back(i);
        }

        inline void update(const uint32_t new_m_a, const uint32_t new_m_b, const uint32_t curr_exp, const size_t i) {
            m_a = new_m_a;
            m_b = new_m_b;
            indexes.push_back(i);
            exponents.push_back(curr_exp);
        }

        bool operator<(const merged_range& mr) const {
            if (indexes.size() != mr.indexes.size())
                return indexes.size() > mr.indexes.size();
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

        for (uint64_t i = 0; i < slope_ranges.size(); ++i) {
            const auto& [a, b] = slope_ranges[i];

            if(a < 0 || b < 0) { // fix later
                continue;
            }
            assert(a >= 0 && b >= 0);

            auto [sign_a, exp_a, mant_a] = get_components(a);
            auto [sign_b, exp_b, mant_b] = get_components(b);

            range r(exp_a, mant_a, exp_b, mant_b, i);

            if (exp_a == exp_b) {
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

        merged_range curr_intersect(eq_exp_ranges[0].m_a,
                                     eq_exp_ranges[0].m_b,
                                     eq_exp_ranges[0].e_a, // notice exp_a = exp_b
                                     eq_exp_ranges[0].idx); 

        for(size_t i = 1; i < eq_exp_ranges.size(); ++i) {
            const uint32_t range_min = eq_exp_ranges[i].m_a;
            const uint32_t range_max = eq_exp_ranges[i].m_b;
            const uint32_t range_exp = eq_exp_ranges[i].e_a; // notice exp_a = exp_b
            const size_t range_idx = eq_exp_ranges[i].idx;

            if (range_min > curr_intersect.m_b) {
                eq_exp_merged.push_back(std::move(curr_intersect));
                curr_intersect = merged_range(range_min, range_max,
                                                 range_exp, range_idx);
            } 
            else {
                curr_intersect.update(std::max(curr_intersect.m_a, range_min),
                                        std::min(curr_intersect.m_b, range_max),
                                        range_exp, range_idx);
            }
        }

        eq_exp_merged.push_back(std::move(curr_intersect));

        // iteratively assign the non-jolly ranges to the
        // intersection covering the most ranges.
        // use a sorted array because updates should be infrequent
        std::sort(eq_exp_merged.begin(), eq_exp_merged.end());

        auto update_range = [&](auto it, uint32_t new_m_a, uint32_t new_m_b, uint32_t new_exp, size_t idx) {
            merged_range updated = std::move(*it);
            eq_exp_merged.erase(it);

            updated.update(new_m_a, new_m_b, new_exp, idx);

            auto pos = std::lower_bound(eq_exp_merged.begin(), eq_exp_merged.end(), updated);

            eq_exp_merged.insert(pos, std::move(updated));
        };

        for(const range &range_to_merge : to_merge) {
            bool merged = false;

            for(auto it = eq_exp_merged.begin(); it != eq_exp_merged.end(); ++it) {
                const merged_range& curr_intersection = *it;

                if(auto first_intersect = intersect(
                        range_to_merge.m_a, std::numeric_limits<uint32_t>::max(),
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

            if(!merged) std::cout << "not merged" << std::endl;
        }

        // finalize the assignment
        for(const auto &intersection : eq_exp_merged) {
            assert(intersection.indexes.size() == intersection.exponents.size());
            const uint32_t mant = intersection.m_a;
            for(size_t i = 0; i < intersection.indexes.size(); ++i) {
                const size_t idx = intersection.indexes[i];
                const uint32_t exp = intersection.exponents[i];
                slopes[idx] = build_float(0, exp, mant);
            }
        }

        // assign the mantissae of jolly ranges to the most frequent
        // mantissa of the merged ranges with equal exponent

        uint32_t most_freq_mant = eq_exp_merged.begin()->m_a;

        for(const auto& range : jolly_ranges) {
            uint32_t exponent;

            if (range.e_b > range.e_a + 1) {
                exponent = range.e_a + 1;
            } else if (most_freq_mant <= range.m_b) { 
                exponent = range.e_b;
            } else {
                exponent = range.e_a;
            }

            slopes[range.idx] = build_float(0, exponent, most_freq_mant);
        }

        // sanity check
        for(size_t i = 0; i < slopes.size(); ++i)
            assert(slopes[i] >= slope_ranges[i].first && slopes[i] <= slope_ranges[i].second);

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

    static std::tuple<uint32_t, uint32_t, uint32_t> get_components(const float slope) {
        uint32_t bits;
        std::memcpy(&bits, &slope, sizeof(float));
        uint32_t sign = (bits >> (total_bits - 1)) & 0x1;

        uint32_t exponent = (bits >> mant_bits) & ((1u << exp_bits) - 1);
        uint32_t mantissa = bits & ((uint32_t(1) << mant_bits) - 1);

        return {sign, exponent, mantissa};
    }

    static float build_float(const uint32_t sign, const uint32_t exponent, const uint32_t mantissa) {
        uint32_t bits = 0;

        bits |= (mantissa & ((uint32_t(1) << mant_bits) - 1));
        bits |= (uint32_t(exponent) & ((1u << exp_bits) - 1)) << mant_bits;
        bits |= (uint32_t(sign & 0x1) << (total_bits - 1));

        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }

};
