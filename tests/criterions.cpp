/*
g++ criterions.cpp -LCaberNet/lib -lCaberNet -I CaberNet/include
*/

#include <CaberNet/CaberNet.h>

int main() {

    net::Tensor X({3, 5}, false); X.fill(
        {
            -1.0, 2.0, -0.5, 1.0, 3.0,
            0.5, 1.0, 2.0, -1.0, -2.0,
            2.0, 1.0, -1.0, 0.5, -0.5
        }
    );
    
    net::Subscripts y({3,1}); y.fill({1, 3, 0});
    X = net::function::log_softmax(X,1);
    
    net::criterion::NegativeLogLikelihood criterion(X, y);

    std::cout << criterion.loss() << std::endl;
}


/*

The result shoud be: 1.82998

Equivalent pytorch code:

import torch
import torch.nn.functional as F

# Generate some example data
# In practice, you would replace this with your actual data and targets
X = torch.tensor([[-1.0, 2.0, -0.5, 1.0, 3.0], [0.5, 1.0, 2.0, -1.0, -2.0], [2.0, 1.0, -1.0, 0.5, -0.5]])
y = torch.tensor([1, 3, 0])  # Example target labels

# Compute the negative log likelihood (NLL) loss

X = F.log_softmax(X,1)
X = F.nll_loss(X,y)
print(X)

*/