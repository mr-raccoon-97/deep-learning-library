#ifndef INTERNAL_FUNCTION_LOGSOFTMAX_H
#define INTERNAL_FUNCTION_LOGSOFTMAX_H

#include "../config.h"

namespace internal {

class LogSoftmax {
    public:
    static void inplace(Tensor* input, int axis);
};

}


#endif // INTERNAL_FUNCTION_LOGSOFTMAX_H
