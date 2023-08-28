#ifndef INTERNAL_TYPES_H
#define INTERNAL_TYPES_H

#include <iostream>
#include <vector>

namespace internal::type {

using scalar_type = float;
using size_type = std::size_t;
using shape_type = std::vector<size_type>;
using storage_type = std::vector<scalar_type>;

} // namespace internal::type;

#endif