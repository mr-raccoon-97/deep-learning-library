/*************************************************************************************************\

This is the computational graph class. It is a singleton class that stores the tensors that are
created during the forward pass. The tensors are stored in a buffer and are deleted when the buffer
is flushed. This allows to preallocate the memory for the tensors before the forward pass, allowing
lazy evaluation. This is useful when the forward pass is performed multiple times, for example in
training and testing phases of a neural network.

/*************************************************************************************************/

#ifndef INTERNAL_GRAPH_HPP
#define INTERNAL_GRAPH_HPP

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

#endif // INTERNAL_GRAPH_HPP