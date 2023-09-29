#include "../config.h"
#include "internal_criterions.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

NLLLoss::scalar_type NLLLoss::loss() const {
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> output_map(
            output()->forward()->data(),
            batch_size(),
            number_of_classes()
        );

        scalar_type loss_value = 0;
        for (auto index = 0; index < batch_size(); ++index) {
            loss_value -= output_map(index, targets()->data()[index]);
        }

        return loss_value / static_cast<scalar_type>(batch_size());
    }
}

#endif // USE_EIGEN_BACKEND