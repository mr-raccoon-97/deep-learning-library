#include <CaberNet/CaberNet.h>

struct Encoder : public net::base::Model {
    net::layer::Sequence layers;
    Encoder() {
        layers = {
            new net::layer::Linear(64, 32),
            new net::layer::ReLU(),
            new net::layer::Linear(32, 16),
            new net::layer::ReLU(),
        };
    }

    net::Tensor forward(net::Tensor x) override {
        return layers.forward(x);
    }
};

struct Decoder : public net::base::Model {
    net::layer::Sequence layers;
    Decoder() {
        layers = {
            new net::layer::Linear(16, 32),
            new net::layer::ReLU(),
            new net::layer::Linear(32, 64)
        };
    }

    net::Tensor forward(net::Tensor x) override {
        x = layers.forward(x);
        x = net::function::softmax(x, 1);
        return x;
    }
};

struct Autoencoder : public net::base::Model {
    Encoder encoder;
    Decoder decoder;

    Autoencoder() : encoder(), decoder() {}
    net::Tensor forward(net::Tensor x) override {
        x = encoder.forward(x);
        x = decoder.forward(x);
        return x;
    }
};

int main() {
    Autoencoder model;
}