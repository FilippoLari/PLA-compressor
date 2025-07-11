#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "piecewise_linear_model.hpp"

template<typename X, typename Y, typename Floating, size_t Epsilon>
class PlainPLA {

    static_assert(Epsilon > 0);
    static_assert(std::is_integral_v<X>);
    static_assert(std::is_integral_v<Y>);

    struct Segment;
    using pla_model = OptimalPiecewiseLinearModel<X, Y>;

    std::vector<Segment> segments;

    uint64_t n;

public:

    PlainPLA() = default;

    explicit PlainPLA(const std::vector<Y> &data) : n(data.size()) {
        if(n == 0) [[unlikely]] 
            return;

        segments.reserve(n / (Epsilon * Epsilon));

        auto in_fun = [data](auto i) { return std::pair<X,Y>(i, data[i]); };
        auto out_fun = [&](auto cs) { segments.emplace_back(cs); };

        make_segmentation_par(n, Epsilon, in_fun, out_fun);
    }

    [[nodiscard]] Y predict(const X &v) const {
        const auto it = std::prev(std::upper_bound(segments.begin(), segments.end(), v));
        return (*it)(v);
    }

    inline size_t size() const {
        return segments.size() * sizeof(Segment) * CHAR_BIT;
    }

    inline size_t bps() const {
        return double(size()) / double(segments.size());
    }

};

template<typename X, typename Y, typename Floating, size_t Epsilon>
struct PlainPLA<X, Y, Floating, Epsilon>::Segment {
    X x;
    Floating slope;
    Y intercept;

    Segment() = default;

    Segment(X x, Floating slope, Y intercept) : x(x), slope(slope), intercept(intercept) {};

    explicit Segment(const typename OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment &cs)
        : x(cs.get_first_x()) {
        auto[cs_slope, cs_intercept, _] = cs.get_floating_point_segment(x, 0);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to uint64");
        slope = cs_slope;
        intercept = cs_intercept;
    }

    friend inline bool operator<(const Segment &s, const X &v) { return s.x < v; }
    friend inline bool operator<(const X &v, const Segment &s) { return v < s.x; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.x < t.x; }

    operator X() { return x; };

    inline Y operator()(const X &v) const {
        Y pos;
        if constexpr (std::is_same_v<X, int64_t> || std::is_same_v<X, int32_t>)
            pos = size_t(slope * double(std::make_unsigned_t<X>(v) - x));
        else
            pos = size_t(slope * double(v - x));
        return pos + intercept;
    }
};
