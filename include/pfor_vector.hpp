#pragma once

#include <unordered_map>
#include <vector>

#include "sdsl/rank_support_v5.hpp"
#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

/**
 * A compressed sequence inspired by the PForDelta encoding.
 * It stores the distinct symbols in the sequence sorted by frequency.
 * A separate array stores indexes inside the symbol table. This array is 
 * splitted into two parts: one storing indexes that can be represented 
 * using b bist, and the other storing the remaining indexes. A bit vector keeps 
 * track of this partitioning. b is automatically selected to minimize the space usage.
 * 
 * Space: t*log(u) + b*n_b + (n - n_b)*log(t) + n + o(n) bits.
 *          - u is the largest value in the sequence
 *          - t is the number of distinct values
 *          - b is the number of bits allocated for each index inside the symbol table
 *          - n_b is the number of indexes that can be represented using b bits
 * 
 * Access time: O(1)
 * 
 * TODO: 1. Based on the number of escaped indexes adaptively choose between Elias-Fano
 *          or a plain bit vector. Otherwise, always use Elias-Fano, but adaptively 
 *          negate the bit vector in such a way to minimize the density of 1s.
 */
template<typename T>
class pfor_vector {

    sdsl::int_vector<> symbol_table;

    sdsl::int_vector<> sequence;
    sdsl::int_vector<> escaped_sequence;

    sdsl::bit_vector escaped;
    sdsl::rank_support_v5<> rank_escaped;

    static constexpr int64_t code_len_tab[64] = {
        0UL, 1UL, 3UL, 7UL, 15UL, 31UL, 63UL, 127UL, 
        255UL, 511UL, 1023UL, 2047UL, 4095UL, 8191UL, 16383UL, 32767UL, 
        65535UL, 131071UL, 262143UL, 524287UL, 1048575UL, 2097151UL, 4194303UL, 8388607UL, 
        16777215UL, 33554431UL, 67108863UL, 134217727UL, 268435455UL, 536870911UL, 1073741823UL, 2147483647UL, 
        4294967295UL, 8589934591UL, 17179869183UL, 34359738367UL, 68719476735UL, 137438953471UL, 274877906943UL, 549755813887UL, 
        1099511627775UL, 2199023255551UL, 4398046511103UL, 8796093022207UL, 17592186044415UL, 35184372088831UL, 70368744177663UL, 140737488355327UL, 
        281474976710655UL, 562949953421311UL, 1125899906842623UL, 2251799813685247UL, 4503599627370495UL, 9007199254740991UL, 18014398509481983UL, 36028797018963967UL, 
        72057594037927935UL, 144115188075855871UL, 288230376151711743UL, 576460752303423487UL, 1152921504606846975UL, 2305843009213693951UL, 4611686018427387903UL, 9223372036854775807UL, 
    };

    size_t n;

public:

    pfor_vector() = default;

    template<class container>
    explicit pfor_vector(const container &data) : n(data.size()) {
        if(n == 0) [[unlikely]]
            return;

        std::unordered_map<T, uint64_t> symbols_frequency;
        
        for(const T &symbol : data)
            symbols_frequency[symbol]++;

        std::multimap<uint64_t, T, std::greater<>> sorted_symbols;

        for(const auto &[symbol, frequency] : symbols_frequency) {
            sorted_symbols.emplace(frequency, symbol);
        }

        symbol_table = sdsl::int_vector<>(sorted_symbols.size());
        std::unordered_map<T, size_t> symbols_rank;
        size_t i = 0;

        for(const auto &[_, symbol] : sorted_symbols) {
            symbols_rank[symbol] = i;
            symbol_table[i++] = symbol;
        }

        std::vector<size_t> tmp_sequence;
        std::vector<size_t> tmp_escaped_seq;
        escaped = sdsl::bit_vector(n + 1, 0);

        partition_sequence(data, symbols_rank, tmp_sequence, tmp_escaped_seq);

        sdsl::util::bit_compress(symbol_table);

        sequence = sdsl::int_vector<>(tmp_sequence.size());
        std::copy(tmp_sequence.begin(), tmp_sequence.end(), sequence.begin());
        sdsl::util::bit_compress(sequence);

        escaped_sequence = sdsl::int_vector<>(tmp_escaped_seq.size());
        std::copy(tmp_escaped_seq.begin(), tmp_escaped_seq.end(), escaped_sequence.begin());
        sdsl::util::bit_compress(escaped_sequence);

        sdsl::util::init_support(rank_escaped, &escaped);
    }

    pfor_vector& operator=(const pfor_vector &pfv) {
        symbol_table = pfv.symbol_table;
        sequence = pfv.sequence;
        escaped_sequence = pfv.escaped_sequence;
        escaped = pfv.escaped;
        sdsl::util::init_support(rank_escaped, &escaped);
        return *this;
    }

    const T operator[](const size_t i) const {
        size_t escaped_before = rank_escaped(i);
        size_t symbol_index;

        if (escaped[i]) [[unlikely]] {
            symbol_index = escaped_sequence[escaped_before];
        } else [[likely]] {
            symbol_index = sequence[i - escaped_before];
        }

        return symbol_table[symbol_index];
    }

    uint64_t size() const {
        return  (sdsl::size_in_bytes(symbol_table) +
                sdsl::size_in_bytes(sequence) + sdsl::size_in_bytes(escaped_sequence) +
                sdsl::size_in_bytes(escaped) + sdsl::size_in_bytes(rank_escaped) + 
                sizeof(n)) * CHAR_BIT;
    }

private:

    /**
     * Compute the value of b that minimizes the space usage 
     * s(b) = b*n_b - n_b*log(t). This function is unimodal,
     * i.e., first decreasing and then increasing. the optimal b can be
     * found using a ternary search in O(nloglogn) time.
     * 
     * TODO: avoid scanning the whole input inside compute_unescaped.
     *       If b increases then the number of indexes that can be represented
     *       with b - 1 bits can still be represented with b bits. Just check
     *       the ones that were not representable in the previous iteration.
     */
    template<class container>
    void partition_sequence(const container &data, std::unordered_map<T, size_t> &symbols_rank,
                                std::vector<size_t> &sequence, std::vector<size_t> &escaped_seq) {
        auto compute_unescaped = [&](const uint8_t b) -> size_t {
            size_t unescaped = 0;
            for(const T &element : data) {
                if(symbols_rank[element] <= code_len_tab[b]) unescaped++;
            }
            return unescaped;
        };
        
        auto space = [&](const uint8_t b, const size_t unescaped) -> double {
            size_t escaped = n - unescaped;
            return escaped * std::log2(symbols_rank.size()) + unescaped * b;
        };

        uint8_t lo = 1, hi = 64;

        while(hi - lo > 3) {
            uint8_t mid1 = lo + (hi - lo) / 3;
            uint8_t mid2 = hi - (hi - lo) / 3;

            if (space(mid1, compute_unescaped(mid1)) < space(mid2, compute_unescaped(mid2))) {
                hi = mid2;
            } else {
                lo = mid1;
            }
        }

        uint8_t opt_b = lo;
        double min_val = space(opt_b, compute_unescaped(opt_b));
        for(uint8_t b = lo + 1; b <= hi; ++b) {
            double curr_val = space(b, compute_unescaped(b));
            if (curr_val < min_val) {
                min_val = curr_val;
                opt_b = b;
            }
        }

        for(size_t k = 0; k < n; ++k) {
            const size_t curr_code = symbols_rank[data[k]];
            if(curr_code > code_len_tab[opt_b]) [[unlikely]] {
                escaped_seq.push_back(curr_code);
                escaped[k] = 1;
            } else [[likely]] {
                sequence.push_back(curr_code);
            }
        }
    }

};