#pragma once

#include <unordered_map>
#include <type_traits>
#include <optional>
#include <cstdint>
#include <climits>
#include <vector>
#include <tuple>
#include <set>
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
        uint64_t freq;
        
        merged_range(uint32_t m_a, uint32_t m_b, uint64_t freq) : m_a(m_a), m_b(m_b), freq(freq) {}

        bool operator<(const merged_range& mr) const {
            if (freq != mr.freq)
                return freq > mr.freq;
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

            auto [sign_a, exp_a, mant_a] = get_components(a);
            auto [sign_b, exp_b, mant_b] = get_components(b);

            assert(sign_a >= 0 && sign_b >= 0); // only positive values

            range r(exp_a, mant_a, exp_b, mant_b, i);

            if (exp_a == exp_b)
                eq_exp_ranges.push_back(r);
            else if (exp_b > exp_a + 1 || mant_b >= mant_a)
                jolly_ranges.push_back(r);
            else
                to_merge.push_back(r);
        }

        // sort ranges by mantissa for sweep-line merging
        std::sort(eq_exp_ranges.begin(), eq_exp_ranges.end());

        std::vector<merged_range> eq_exp_merged;

        uint32_t curr_min = eq_exp_ranges[0].m_a;
        uint32_t curr_max = eq_exp_ranges[0].m_b;
        uint64_t curr_freq = 1, first_idx = 0;

        auto finalize_group = [&](uint64_t end_idx) {
            for (uint64_t k = first_idx; k < end_idx; ++k) {
                slopes[eq_exp_ranges[k].idx] =
                    build_float(0, eq_exp_ranges[k].e_a, curr_min);
            }
            eq_exp_merged.emplace_back(curr_min, curr_max, curr_freq);
        };

        for (uint64_t i = 1; i < eq_exp_ranges.size(); ++i) {
            const uint32_t range_min = eq_exp_ranges[i].m_a;
            const uint32_t range_max = eq_exp_ranges[i].m_b;

            if (range_min > curr_max) {
                finalize_group(i);
                curr_min  = range_min;
                curr_max  = range_max;
                curr_freq = 1;
                first_idx = i;
            } 
            else {
                curr_min = std::max(curr_min, range_min);
                curr_max = std::min(curr_max, range_max);
                ++curr_freq;
            }
        }

        finalize_group(eq_exp_ranges.size());

        // assign the mantissae of jolly ranges to the most frequent
        // mantissa of the merged ranges with equal exponent
        std::sort(eq_exp_merged.begin(), eq_exp_merged.end());

        uint32_t most_freq_mant = eq_exp_merged[0].m_a;
        eq_exp_merged[0].freq += jolly_ranges.size();

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

        std::multiset<merged_range> freq_sorted_ranges(eq_exp_merged.begin(), eq_exp_merged.end());
        uint32_t freq = freq_sorted_ranges.begin()->freq;

        std::cout << "most frequent: " << freq << std::endl;

        auto update_range = [&](auto it, uint32_t new_m_a, uint32_t new_m_b) {
            merged_range updated = *it;
            updated.m_a = new_m_a;
            updated.m_b = new_m_b;
            ++updated.freq;

            freq_sorted_ranges.erase(it);
            freq_sorted_ranges.insert(updated);
        };

        for (const range& range_to_merge : to_merge) {
            bool merged = false;

            for (auto it = freq_sorted_ranges.begin(); it != freq_sorted_ranges.end(); ++it) {
                const merged_range& existing_range = *it;

                if (auto first_intersect = intersect(
                        range_to_merge.m_a, std::numeric_limits<uint32_t>::max(),
                        existing_range.m_a, existing_range.m_b)) {

                    auto [a, b] = *first_intersect;
                    update_range(it, a, b);
                    merged = true;
                    break;
                }

                if (auto second_intersect = intersect(
                        0, range_to_merge.m_b,
                        existing_range.m_a, existing_range.m_b)) {

                    auto [a, b] = *second_intersect;
                    update_range(it, a, b);
                    merged = true;
                    break;
                }
            }

            if (!merged)
                freq_sorted_ranges.insert(merged_range(range_to_merge.m_a, range_to_merge.m_b, 1));
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
