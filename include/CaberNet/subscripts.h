#pragma once
#include <iostream>
#include <vector>
#include <memory>

// The bad news is that I couldn't make the Tensor of integer class work, tried with std::variant and std::visit
// but I think that it just won't be possible.
// Te good news is that we may not need a Tensor of integers, since it will have a completly different concern
// than the Tensor of floats, I would be a bad idea to mix types. This is not python.
// If you came up with a better name idea than subscripts, please let me know.

namespace internal { template<typename T> class Array; }

namespace net {

class Subscripts {
    public:
    using value_type = int;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;

    using iterator = std::vector<int>::iterator;
    using const_iterator = std::vector<int>::const_iterator;

    Subscripts() = default;
    Subscripts(std::shared_ptr<internal::Array<int>> subscripts);
    Subscripts(shape_type shape);

    void reshape(shape_type shape);
    void fill(value_type value);
    void fill(std::vector<value_type> values);

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    pointer data();
    const_pointer data() const;
    shape_type shape() const;
    size_type rank() const;

    friend std::ostream& operator<<(std::ostream& ostream, const Subscripts& subscripts);

    private:
    std::shared_ptr<internal::Array<int>> subscripts_;
};

} // namespace net