#ifndef INTERNAL_BUFFER_HPP
#define INTERNAL_BUFFER_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_tensor.hpp"

namespace internal {

class Expression;

class Graph {
    public:
    static Graph& instance() { static Graph buffer; return buffer; }
    static void add(std::shared_ptr<Tensor> tensor) { instance().buffer_.push_back(tensor); }
    static void flush() { instance().buffer_.clear(); }

    private:
    Graph() = default;
    ~Graph() = default;
    Graph(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;
    Graph& operator=(const Graph&) = delete;
    std::vector<std::shared_ptr<Tensor>> buffer_;
};

} // namespace internal

#endif // INTERNAL_BUFFER_HPP