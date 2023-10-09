#include "CaberNet.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::ElementsAre;

TEST(optimizer, sgd) {

    net::Tensor<float> X({2,2}, true); X.fill(1);
    net::Tensor<float> I({2,2}); I.fill(1);
    X.backward(I);

    net::optimizer::SGD optimizer(0.1);
    optimizer.add_parameter(X.internal());
    optimizer.step();

    EXPECT_THAT(X, ElementsAre(0.9, 0.9, 0.9, 0.9));
}