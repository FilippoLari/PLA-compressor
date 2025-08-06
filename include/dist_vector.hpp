#pragma once

#include <unordered_map>
#include <vector>

#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

/**
 * A compressed sequence storing only the distinct
 * elements plus an additional array to perform 
 * random access. If t is very small, the space usage
 * is comparable with huffman coding, but the access time
 * is an order of magnitude faster.
 * 
 * Space: nlog(t) + tlog(|S|)
 *      - n is the sequence length
 *      - t is the number of distinct elements
 *      - |S| is the alphabet size
 * 
 * Access time: O(1) 
 */
template<typename T>
class dist_vector {

    sdsl::int_vector<> distinct;
    sdsl::int_vector<> index_mapping;

public:

    dist_vector() = default;

    template<class container>
    explicit dist_vector(const container &data) {
        index_mapping = sdsl::int_vector<>(data.size());
        std::unordered_map<T, uint64_t> dictionary;
        std::vector<T> tmp_distinct;

        for(uint64_t i = 0; i < data.size(); ++i) {
            if(dictionary.count(data[i]) == 0) {
                dictionary[data[i]] = tmp_distinct.size();
                tmp_distinct.push_back(data[i]);
            }
            
            index_mapping[i] = dictionary[data[i]];
        }

        sdsl::util::bit_compress(index_mapping);

        distinct = sdsl::int_vector<>(tmp_distinct.size());
        std::copy(tmp_distinct.begin(), tmp_distinct.end(), distinct.begin());
        sdsl::util::bit_compress(distinct);
    }

    dist_vector& operator=(const dist_vector &dv) {
        distinct = dv.distinct;
        index_mapping = dv.index_mapping;
        return *this;
    }

    const T operator[](const uint64_t i) const {
       return distinct[index_mapping[i]];
    }

    uint64_t size() const {
        return  (sdsl::size_in_bytes(distinct) +
                + sdsl::size_in_bytes(index_mapping)) * CHAR_BIT;
    }

    inline std::map<T, size_t> get_frequencies() const {
        std::map<T, size_t> frequencies;
        for(const auto &index : index_mapping)
            frequencies[distinct[index]]++;
        return frequencies;
    }
};