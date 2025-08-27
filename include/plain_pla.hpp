#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "piecewise_linear_model.hpp"

template<typename X, typename Y, typename Floating, bool Indexing = true>
class PlainPLA {

    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    struct Segment;
    using pla_model = OptimalPiecewiseLinearModel<X, Y>;

    std::vector<Segment> segments;

    uint64_t epsilon;

    size_t n;
    X u;

public:

    PlainPLA() = default;

    explicit PlainPLA(const std::vector<std::conditional_t<Indexing, X, Y>>& data,
                         const uint64_t epsilon) : n(data.size()), u(data.back()) {
        if(n == 0) [[unlikely]] 
            return;

        segments.reserve(n / (epsilon * epsilon));

        auto in_fun = [data](auto i) { 
            if constexpr (Indexing)
                return std::pair<X, Y>(data[i], i); 
            else
                return std::pair<X, Y>(i, data[i]);
        };

        auto out_fun = [&](auto cs) { segments.emplace_back(cs); };

        make_segmentation_par(n, epsilon, in_fun, out_fun);
    }

    [[nodiscard]] int64_t predict(const X &v) const {
        const auto it = std::prev(std::upper_bound(segments.begin(), segments.end(), v));
        return (*it)(v);
    }

    inline size_t size() const {
        return segments.size() * sizeof(Segment) * CHAR_BIT;
    }

    inline double bps() const {
        return double(size()) / double(segments.size());
    }

    inline size_t get_segments() const {
        return segments.size();
    }

    inline double lower_bound_indexing() const {
        const double l = static_cast<double>(segments.size());

        double sum = 0.0;
        double prev = static_cast<double>(segments[0].x);

        for(size_t i = 1; i < segments.size(); ++i) {
            double curr = static_cast<double>(segments[i].x);
            sum += std::log2(curr - prev + 1.0);
            prev = curr;
        }

        return (l * std::log2(static_cast<double>(u - l * (2.0 * epsilon - 1.0)) / l) +
               (l - 1.0) * std::log2(static_cast<double>(n - l*(2.0 * epsilon - 1.0) - 1.0) / (l - 1.0)) +
               2.0 * l * std::log2(2.0 * epsilon + 1.0) + 
                std::log2(sum));
    }

    inline double lower_bound_compression(const std::vector<X>& data) const {
        const double l = static_cast<double>(segments.size());

        double sum = 0.0;
        double prev = static_cast<double>(data[segments[0].x]);

        for(size_t i = 1; i < segments.size(); ++i) {
            double curr = static_cast<double>(data[segments[i].x]);
            sum += std::log2(curr - prev + 1.0);
            prev = curr;
        }

        return (l * std::log2(static_cast<double>(u - l) / l) +
               (l - 1.0) * std::log2(static_cast<double>(n - l - 1.0) / (l - 1.0)) +
               2.0 * l * std::log2(2.0 * epsilon + 1.0) + 
                std::log2(sum));
    }

};

template<typename X, typename Y, typename Floating, bool Indexing>
struct PlainPLA<X, Y, Floating, Indexing>::Segment {
    X x;
    Floating slope;
    Y intercept;

    Segment() = default;

    Segment(X x, Floating slope, Y intercept) : x(x), slope(slope), intercept(intercept) {};

    explicit Segment(const typename OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment &cs)
        : x(cs.get_first_x()) {
        auto[cs_slope, cs_intercept, _] = cs.get_floating_point_segment(x);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to uint64");
        slope = cs_slope;
        intercept = cs_intercept;
    }

    friend inline bool operator<(const Segment &s, const X &v) { return s.x < v; }
    friend inline bool operator<(const X &v, const Segment &s) { return v < s.x; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.x < t.x; }

    operator X() { return x; };

    inline int64_t operator()(const X &v) const {
        if constexpr (std::is_same_v<X, int64_t> || std::is_same_v<X, int32_t>)
            return static_cast<int64_t>(slope * double(static_cast<std::make_unsigned_t<X>>(v) - x) + intercept);
        else
            return static_cast<int64_t>(slope * double(v - x) + intercept);
    }
};
