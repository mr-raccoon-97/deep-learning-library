#include <CaberNet.h>

struct Network : public net::Model<Network> {
    Network() {
        layers.configure_optimizer(optimizer);
    }

    net::layer::Sequence layers {
        net::layer::Linear(784, 128),
        net::layer::ReLU(),
        net::layer::Linear(128, 10),
        net::layer::LogSoftmax(/*axis*/ 1)
    };

    net::Tensor<float> forward(net::Tensor<float> x) {
        return layers(x);
    }

    net::optimizer::SGD optimizer {/*learning rate*/ 0.1};
};

int main() {
    net::Dataset dataset(64, false);

    dataset.read_targets("data/train-labels.idx1-ubyte");
    dataset.read_features("data/train-images.idx3-ubyte");

    net::Tensor<int> targets({64});
    net::Tensor<float> input({64,784}, false, true);

    Network model;

    net::Tensor<float> output({64,10});
    output = model(input);

    net::criterion::NLLLoss criterion(output, targets);

    for(int epoch = 0; epoch < 10; ++epoch) {

        std::cout << "Epoch: " << epoch + 1 << std::endl;
        
        for(int batch = 0; batch < dataset.length(); ++batch) {
            input.copy(dataset.features()[batch].internal()); // I will fix this in the future so it will be prettier and without copies.
            targets.copy(dataset.targets()[batch].internal());

            std::cout << "loss" << criterion.loss() << std::endl;
            criterion.backward();
            model.optimizer.step();   
        }
        
    }
}
