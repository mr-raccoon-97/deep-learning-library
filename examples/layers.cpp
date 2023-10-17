/*

To run this code build the library following the instructions in the .github folder.

then compile this file with:

cmake . -DCABERNET_BUILD_EXAMPLES=ON
cmake --build . --target cabernet-examples-layers

*/

#include <CaberNet.h>

struct Autoencoder : public net::Model<Autoencoder> {

    Autoencoder() {
        encoder.configure_optimizer(std::dynamic_pointer_cast<net::base::Optimizer>(encoder_optimizer));
        decoder.configure_optimizer(std::dynamic_pointer_cast<net::base::Optimizer>(decoder_optimizer));
    }

    net::layer::Sequence encoder {
        net::layer::Linear(784, 128, net::initializer::He),
        net::layer::ReLU(),
        net::layer::Linear(128, 64, net::initializer::He),
    };

    net::layer::Sequence decoder {
        net::layer::Linear(64, 128, net::initializer::He),
        net::layer::ReLU(),
        net::layer::Linear(128, 784, net::initializer::He),
        net::layer::LogSoftmax(/*axis*/ 1)
    };

    net::Tensor<float> forward(net::Tensor<float> x) {
        x = encoder(x);
        x = decoder(x);
        return x;
    }
    
    void step() {
        encoder_optimizer->step();
        decoder_optimizer->step();
    }
    /* you can add different optimizers to different layers
    or the same, doesn't matter, the optimizer has a shared pointer
    to it's implementation so you can pass instances of it with value
    semantics without making deep copies */

    std::shared_ptr<net::optimizer::SGD> encoder_optimizer = std::make_shared<net::optimizer::SGD>(/*learning rate*/ 0.1);
    std::shared_ptr<net::optimizer::SGD> decoder_optimizer = std::make_shared<net::optimizer::SGD>(/*learning rate*/ 0.2);
};

int main() {
    Autoencoder model;
    net::Tensor<float> x({1, 784}); x.fill(net::initializer::He);
    net::Tensor<float> y = model(x);
    y.perform();
    std::cout << y;

    model.step();
    return 0;
}
