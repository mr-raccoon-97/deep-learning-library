#include <CaberNet.h>

struct Autoencoder : public net::Model<Autoencoder> {

    Autoencoder() = default;

    net::layer::Sequence encoder {
        net::layer::Linear(784, 128, net::initializer::He),
        net::layer::ReLU(),
        net::layer::Linear(128, 64, net::initializer::He),
    };

    net::layer::Sequence decoder {
        net::layer::Linear(64, 128, net::initializer::He),
        net::layer::ReLU(),
        net::layer::Linear(128, 784, net::initializer::He),
        net::layer::LogSoftmax(1/*axis*/)
    };

    net::Tensor forward(net::Tensor x) {
        x = encoder(x);
        x = decoder(x);
        return x;
    }

};

int main() {
    Autoencoder model;
    net::Tensor x({1, 784}); x.fill(net::initializer::He);
    net::Tensor y = model(x);
    y.perform();
    std::cout << y;
}
