#ifndef INTERNAL_BUFFER_HPP
#define INTERNAL_BUFFER_HPP

#include <iostream>
#include <vector>

namespace internal {

class Expression;

class Buffer {
    public:

    static Buffer& instance() {
        static Buffer instance;
        return instance;
    }
    
    static void flush() {
        auto& buffer = instance()._buffer;
        for(auto& element : buffer) delete element;
        buffer.clear(); 
    }

    static void cache(Expression* data) {
        auto& buffer = instance()._buffer;
        buffer.push_back(data);
    }

    ~Buffer() { flush(); }
    
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