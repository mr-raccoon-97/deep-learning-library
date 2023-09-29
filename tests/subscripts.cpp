#include "CaberNet.h"
#include <gtest/gtest.h>

TEST(subscripts, fill) {
    net::Subscripts y({2, 3, 4});
    y.fill(1);

    // ASSERT_EQ(0, y);
}
