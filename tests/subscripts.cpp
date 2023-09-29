#include "CaberNet.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::ElementsAre;

TEST(subscripts, fill) {
    net::Subscripts y({2, 3, 4});
    y.fill(1);

    EXPECT_THAT(y, ElementsAre(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
}
