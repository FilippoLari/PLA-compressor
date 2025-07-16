#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include <cassert>
#include <climits>

#include "sdsl/int_vector.hpp"
#include "sdsl/util.hpp"

template <typename T>
class HuffmanTree {

    struct Node;

    using NodePtr = std::shared_ptr<Node>;

    struct Node {
        T key;
        uint64_t freq;
        NodePtr next, left, right;
        Node(uint64_t freq) : key(), freq(freq), next(), left(), right() {}
        Node(T key, uint64_t freq)
            : key(key), freq(freq), next(), left(), right() {}
    };

    uint64_t number_of_alphabet;
    NodePtr root;

public:

    HuffmanTree() = default;

    explicit HuffmanTree(const std::map<T, uint64_t> &alphabet_count) : number_of_alphabet(alphabet_count.size()) {
        std::vector<NodePtr> leaves;
        leaves.reserve(number_of_alphabet);
        for (auto &&[key, freq] : alphabet_count) {
            leaves.push_back(std::make_shared<Node>(key, freq));
        }
        std::sort(leaves.begin(), leaves.end(),
                [](const NodePtr &a, const NodePtr &b) -> bool {
                    return a->freq < b->freq;
                });
        for (int i = 0; i < number_of_alphabet - 1; i++) {
            leaves[i]->next = leaves[i + 1];
        }
        NodePtr insert_pos = leaves[0];
        NodePtr look_pos = leaves[0];
        while (look_pos->next != nullptr) {
            NodePtr n =
                std::make_shared<Node>(look_pos->freq + look_pos->next->freq);
            n->left = look_pos;
            n->right = look_pos->next;
            while (insert_pos->next != nullptr &&
                insert_pos->next->freq <= n->freq) {
                insert_pos = insert_pos->next;
            }
            n->next = insert_pos->next;
            insert_pos->next = n;
            look_pos = look_pos->next->next;
        }
        root = look_pos;

        auto delete_next = [](auto self, NodePtr n) -> void {
            n->next = nullptr;
            if (n->left != nullptr) {
                self(self, n->left);
            }
            if (n->right != nullptr) {
                self(self, n->right);
            }
        };
        delete_next(delete_next, root);
    }

    std::vector<std::pair<T, uint64_t>> compute_length() {
        std::vector<std::pair<T, uint64_t>> res;
        res.reserve(number_of_alphabet);

        auto dfs = [&](auto self, NodePtr n, uint64_t d = 0) -> void {
            if (n->left == nullptr) {
                res.push_back({n->key, d});
            } else {
                self(self, n->left, d + 1);
                self(self, n->right, d + 1);
            }
        };
        dfs(dfs, root);

        return res;
    }
};

template<typename T, uint32_t dens = 128>
class huffman_vector {

    sdsl::bit_vector code_sequence;

    sdsl::int_vector<> sample_pointers;

    sdsl::int_vector<> alphabet;
    sdsl::int_vector<> first_index, first_code;

    uint64_t alphabet_size, maximum_code_length;

public:

    huffman_vector() = default;

    template<class container>
    explicit huffman_vector(const container &data) {
        std::map<T, uint64_t> symbol_frequencies;
        for(const T &symbol : data) symbol_frequencies[symbol]++;

        alphabet_size = symbol_frequencies.size();

        HuffmanTree<T> ht(symbol_frequencies);
        auto symbols_to_code_lenghts = ht.compute_length();
        sort(symbols_to_code_lenghts.begin(), symbols_to_code_lenghts.end(),
            [](const std::pair<T, uint64_t> &a, const std::pair<T, uint64_t> &b)
                -> bool { return a.second < b.second; });

        // compute the symbol table sorted by code length and the
        // final length of the concatenated sequence of symbols
        uint64_t code_sequence_length = 0;
        alphabet = sdsl::int_vector<>(symbols_to_code_lenghts.size());
        for(uint64_t i = 0; i < symbols_to_code_lenghts.size(); ++i) {
            const T symbol = symbols_to_code_lenghts[i].first;
            alphabet[i] = symbol;
            code_sequence_length += symbol_frequencies[symbol] * symbols_to_code_lenghts[i].second; 
        }

        // map symbol to index of sorted symbols by length,
        // it is used to speed up the code sequence construction
        std::map<T, uint32_t> inverse_alphabet;
        for(uint64_t i = 0; i < alphabet.size(); ++i)
            inverse_alphabet[alphabet[i]] = i;

        sdsl::util::bit_compress(alphabet);

        maximum_code_length = symbols_to_code_lenghts.back().second;
        assert(maximum_code_length <= 64);

        std::vector<uint64_t> codes;
        codes.reserve(alphabet_size);
        codes.push_back(0);
        for(int i = 1; i < alphabet_size; i++) {
            codes.push_back((codes.back() + 1)
                            << (symbols_to_code_lenghts[i].second -
                                symbols_to_code_lenghts[i - 1].second));
        }

        // todo: avoid writing a single bit at a time
        code_sequence = sdsl::bit_vector(code_sequence_length);
        uint64_t offset = 0;

        uint64_t samples = (data.size() + dens - 1) / dens;
        sample_pointers = sdsl::int_vector<>(samples + 1);
        uint64_t samples_idx = 0;

        for(uint64_t i = 0; i < data.size(); ++i) {

            // sample a starting position every dens elements
            if((i % dens) == 0) [[unlikely]] {
                sample_pointers[samples_idx] = offset;
                samples_idx++;
            }

            const uint64_t symbol_index = inverse_alphabet[data[i]];
            const uint64_t code = codes[symbol_index];
            const uint64_t len = symbols_to_code_lenghts[symbol_index].second;

            for(uint64_t i = 0; i < len; ++i) // notice: from lsb to msb
                code_sequence[offset + i] = (code & (1UL << i))? 1 : 0;

            offset += len;
        }

        // todo: fix this
        if(*sample_pointers.end() == 0) sample_pointers.resize(sample_pointers.size() - 1);

        sdsl::util::bit_compress(sample_pointers);

        /*for(uint64_t i = 0; i < code_sequence.size(); ++i)
            std::cout << code_sequence[i];
        std::cout << std::endl;

        for(uint64_t i = 0; i< sample_pointers.size(); ++i)
            std::cout << "sample pos: " << sample_pointers[i] << std::endl;*/

        first_index = sdsl::int_vector<>(maximum_code_length + 1, alphabet_size);
        first_code = sdsl::int_vector<>(maximum_code_length + 1);
        for(int i = alphabet_size - 1; i >= 0; i--) {
            first_index[symbols_to_code_lenghts[i].second] = i;
            first_code[symbols_to_code_lenghts[i].second] = codes[i];
        }
        for(int l = maximum_code_length - 1; l >= 1; l--) {
            if (first_index[l] == alphabet_size) {
                first_index[l] = first_index[l + 1];
                first_code[l] = first_code[l + 1] >> 1;
            }
        }

        sdsl::util::bit_compress(first_index);
        sdsl::util::bit_compress(first_code);
    }

    huffman_vector& operator=(const huffman_vector &hv) {
        code_sequence = hv.code_sequence;
        sample_pointers = hv.sample_pointers;
        alphabet = hv.alphabet;
        first_index = hv.first_index;
        first_code = hv.first_code;
        alphabet_size = hv.alphabet_size;
        maximum_code_length = hv.maximum_code_length;
        return *this;
    }

    const T operator[](const uint64_t i) const {
        const uint64_t sample_pos = i / dens;
        uint64_t curr_pos = sample_pointers[sample_pos];
        uint64_t len;

        //std::cout << "curr pos: " << curr_pos << " sample pos: " << sample_pos << std::endl;

        for(uint64_t j = sample_pos * dens; j <= i; ++j) {
            len = get_code_length(curr_pos);
            curr_pos += len;
            //std::cout << "len: " << len << " curr pos: " << curr_pos << std::endl;
        }

        //std::cout << "i: " << i << std::endl;
        //std::cout << "len: " << len << " off: " << curr_pos - len << std::endl;

        const uint64_t code = code_sequence.get_int(curr_pos - len, len);

        return decode(len, code);
    }

    uint64_t size() const {
        return  code_sequence.bit_size() +
                + (sdsl::size_in_bytes(sample_pointers) +
                + sdsl::size_in_bytes(first_index) + sdsl::size_in_bytes(first_code) +
                + sdsl::size_in_bytes(alphabet)) * CHAR_BIT;
    }

private:

    inline T decode(const uint64_t code_length, const uint64_t code) const {
        return alphabet[first_index[code_length] + code - first_code[code_length]];
    }

    uint64_t get_code_length(const uint64_t i) const {
        uint64_t next = code_sequence.get_int(i, maximum_code_length);
        int l = 0, r = maximum_code_length + 1;
        int m;
        while (r - l > 1) {
            m = (l + r) / 2;
            if ((first_code[m] << (maximum_code_length - m)) <= next) {
                l = m;
            } else {
                r = m;
            }
        }
        return l;
    }
};