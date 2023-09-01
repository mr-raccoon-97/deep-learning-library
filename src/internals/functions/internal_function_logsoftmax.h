#ifndef INTERNAL_FUNCTION_LOGSOFTMAX_H
#define INTERNAL_FUNCTION_LOGSOFTMAX_H

#include "../config.h"
#include "../internal_expression.hpp"

namespace internal {

class LogSoftmax : public Expression {
    public:
    static void inplace(Tensor* input, int axis);
};

}


#endif // INTERNAL_FUNCTION_LOGSOFTMAX_H
