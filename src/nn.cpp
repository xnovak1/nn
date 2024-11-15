#include <cmath>
#include <vector>
#include <algorithm>
#include "matrix.hpp"

using namespace std;

#define N_INPUT 784
#define N_HIDDEN 512
#define N_OUTPUT 10
#define LEARNING_RATE 0.1
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

struct Layer {
    int n_input;  // number of input neurons
    int n_output; // number of output neurons
    Matrix weights;
    Matrix biases;
    Matrix output;
};

struct Network {
    Layer hidden;
    Layer output;
};

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

float activation(float x) {
    return relu(x);
    // return sigmoid(x);
}

/**
 * @brief Softmax activation function.
 * 
 * @param input Output of last layer neurons
 * @param size Number of categories
 */
void softmax(vector<float> &input, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }

    for (int i = 0; i < size; i++)
        input[i] = input[i] / sum;
}

/**
 * @brief Returns index of a biggest element.
 * 
 * @param input Vector of floats
 * @return int Index
 */
int argmax(vector<float> input) {
    float max = input[0];
    int max_i = 0;

    for (size_t i = 0; i < input.size(); i++) {
        if (input[i] > max) {
            max = input[i];
            max_i = i;
        }
    }

    return max_i;
}

/**
 * @brief Forward evaluation of a layer.
 * 
 * @param layer Layer
 * @param input Vector of neuron inputs
 * @param output Vector of neuron outputs
 */
void layer_forward(Layer &layer, vector<float> input, vector<float> &output, bool activate) {
    for (int j = 0; j < layer.n_output; j++) {
        output[j] += layer.biases.at(0, j);
        for (int i = 0; i < layer.n_input; i++) {
            output[j] += input[i] * layer.weights.at(i, j);
        }

        if (activate) output[j] = activation(output[j]);
    }
}

/**
 * @brief Single forward pass. Evaluation of input.
 * 
 * @param input Vector of pixels
 * @return int Image classification (0-9)
 */
int forward(Network network, vector<float> input) {
    vector<float> output_hidden;
    vector<float> output_last;

    layer_forward(network.hidden, input, output_hidden, true);
    layer_forward(network.output, output_hidden, output_last, false);
    softmax(output_last, 10);

    return argmax(output_last);
}


void backprop(Matrix labels, Network nn);