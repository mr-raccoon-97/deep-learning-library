#include "CaberNet.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::ElementsAre;

TEST(functions, gradient) {
    /*
    import torch.nn.functional as F

    # Initialize tensors
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=False)
    w = torch.tensor([[1, 2, -3], [4, 5, 6], [7, 8, -9], [10, 11, -12]], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
    I = torch.tensor([[1,1,1,1],[1,1,1,1]], dtype=torch.float32)

    # Perform linear operation using torch.nn.functional.linear
    x = F.linear(x, w, b)
    x = F.relu(x)
    x.backward(I)

    # Print the result
    print(x)
    print(w.grad)
    print(b.grad)

    */
   
    net::Tensor<float> x({2,3}, false); x.fill({1,2,3,4,5,6});
    net::Tensor<float> w({4,3}, true); w.fill({1,2,-3,4,5,6,7,8,-9,10,11,-12});
    net::Tensor<float> b({1,4}, true); b.fill({1,2,3,4});
    net::Tensor<float> I({2,4}, false); I.fill(1);

    x = net::function::linear(x,w,b);
    x = net::function::relu(x);
    x.perform();
    x.backward(I);

    /*
    Results should be:
    [0, 34, 0, 0, 0, 79, 17, 27, ]
    [0, 0, 0, 5, 7, 9, 4, 5, 6, 4, 5, 6, ]
    [0, 2, 1, 1, ]
    */

    EXPECT_THAT(x, ElementsAre(0, 34, 0, 0, 0, 79, 17, 27));
    EXPECT_THAT(w.gradient(), ElementsAre(0, 0, 0, 5, 7, 9, 4, 5, 6, 4, 5, 6));
    EXPECT_THAT(b.gradient(), ElementsAre(0, 2, 1, 1));
}
