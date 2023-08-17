#ifndef INTERNAL_BUFFER_HPP
#define INTERNAL_BUFFER_HPP

#include <iostream>
#include <vector>

namespace internal {

class Buffer {
    public:
    static Buffer& instance() { static Buffer instance; return instance; }
    ~Buffer() { flush(); }
    void flush() { for(auto& element : _buffer) delete element; _buffer.clear(); }
    void operator << (Expression* expression) { _buffer.push_back(expression); }

    private:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::vector<Expression*> _buffer;
};

} // namespace internal

#endif // INTERNAL_BUFFER_HPP