#ifndef INTERNAL_FUNCTION_SOFTMAX_H
#define INTERNAL_FUNCTION_SOFTMAX_H

#include "../config.h"
#include "../internal_expression.hpp"

namespace internal {

class Softmax : public Expression {
    public:
    static void inplace(Tensor* input, int axis);
};

}


#endif // INTERNAL_FUNCTIONS_SOFTMAX_H
