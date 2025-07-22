#pragma once

#include <unordered_map>
#include <vector>

#include "sdsl/int_vector.hpp"
#include "sdsl/sd_vector.hpp"
#include "sdsl/util.hpp"

/**
 * A run-length compressed sequence supporting random
 * access in logarithmic time.
 * 
 * Space: r(log(|S|*n/r) + 2 + o(1)) bits.
 *      - n is the sequence length
 *      - r is the number of runs
 *      - |S| is the alphabet size
 * 
 * Access time: O(min{log(n/r), log(r)}) 
 */
template<typename T>
class rle_vector {

    sdsl::int_vector<> run_heads;

    sdsl::sd_vector<> run_start;
    sdsl::rank_support_sd<> rank_run_start;

public:

    rle_vector() = default;

    template<class container>
    explicit rle_vector(const container &data) {
        if(data.size() == 0) [[unlikely]]
            return;

        sdsl::bit_vector bv_run_start(data.size(), 0);
        std::vector<T> tmp_heads;

        tmp_heads.push_back(data[0]);
        bv_run_start[0] = 1;

        for(uint64_t i = 1; i < data.size(); ++i) {
            if(data[i] != data[i-1]) {
                tmp_heads.push_back(data[i]);
                bv_run_start[i] = 1;
            }
        }

        run_heads = sdsl::int_vector<>(tmp_heads.size());
        std::copy(tmp_heads.begin(), tmp_heads.end(), run_heads.begin());
        sdsl::util::bit_compress(run_heads);

        run_start = sdsl::sd_vector<>(bv_run_start);
        sdsl::util::init_support(rank_run_start, &run_start);
    }

    rle_vector& operator=(const rle_vector &rlv) {
        run_heads = rlv.run_heads;
        run_start = rlv.run_start;
        rank_run_start = rlv.rank_run_start;
        return *this;
    }

    const T operator[](const uint64_t i) const {
       const uint64_t run_idx = rank_run_start(i);
       return run_heads[run_idx];
    }

    uint64_t size() const {
        return  (sdsl::size_in_bytes(run_heads) +
                + sdsl::size_in_bytes(run_start) + sdsl::size_in_bytes(rank_run_start)) * CHAR_BIT;
    }
};