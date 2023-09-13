#ifndef INTERNAL_BUFFER_HPP
#define INTERNAL_BUFFER_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_tensor.hpp"

namespace internal {

class Expression;

class Buffer {
    public:
    static Buffer& instance() { static Buffer buffer; return buffer; }
    static void add(std::shared_ptr<Tensor> tensor) { instance()._buffer.push_back(tensor); }
    static void flush() { instance()._buffer.clear(); }
    
    private:
    Buffer() = default;
    ~Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::vector<std::shared_ptr<Tensor>> _buffer;
};

} // namespace internal

#endif // INTERNAL_BUFFER_HPP