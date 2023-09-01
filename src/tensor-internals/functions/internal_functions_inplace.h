#ifndef INTERNAL_FUNCTIONS_INPLACE_H
#define INTERNAL_FUNCTIONS_INPLACE_H

namespace internal {

class Tensor;

void softmax_inplace(Tensor* input, int axis);
void log_softmax_inplace(Tensor* input, int axis);

}

#endif // INTERNAL_FUNCTIONS_INPLACE_H