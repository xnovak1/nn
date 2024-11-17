#pragma once

#include <vector>
#include "matrix.hpp"

struct Layer {
    int n_input;  // number of input neurons
    int n_output; // number of output neurons
    Matrix weights;
    Matrix biases;
    Matrix outputs;
};

struct Network {
    Layer hidden;
    Layer output;
};

int predict(Network network, std::vector<float> input);
void train(Network nn, int epochs, int batch_size);