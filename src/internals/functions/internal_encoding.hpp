#ifndef INTERNAL_ENCODING_H
#define INTERNAL_ENCODING_H

#include "../internal_expression.hpp"
#include "../internal_types.h"
#include "../internal_tensor.hpp"

#include <iostream>
#include <vector>

namespace internal {

Tensor one_hot_encode( const std::vector<int>& targets, std::size_t number_of_classes ) {
    type::size_type batch_size = targets.size();
    Tensor encoded_targets({batch_size, number_of_classes});
    for (auto& element : encoded_targets) element = 0;
    for (auto index = 0; index < batch_size; index++) {
        encoded_targets.data()[index * batch_size + targets[index]] = 1;
    }
    return encoded_targets;
}

}

#endif // INTERNAL_ENCODING_H_PP