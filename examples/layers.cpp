#include <CaberNet/CaberNet.h>

struct Encoder : public net::base::Model {
    public:
    Encoder() {
        layers = {
            new net::layer::Linear(64, 32),
            new net::layer::ReLU(),
            new net::layer::Linear(32, 16),
            new net::layer::ReLU(),
        };
    }

    net::Tensor forward(net::Tensor x) {
        return layers.forward(x);
    }

    net::layer::Sequence layers;
};


int main() {
    Encoder encoder;
    net::Tensor input({256, 64}); input.fill(net::initializer::He);
    net::Tensor output = encoder.forward(input);
    output.perform();
    for(auto element : output) std::cout << element;
}