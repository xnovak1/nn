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

/**
 * @brief Single forward pass. Evaluation of input.
 * 
 * @param input Vector of pixels
 * @return int Image classification (0-9)
 */
int forward(vector<float> input);


void backprop(Matrix labels, Network nn){
    Matrix out= nn.output.output;
    


}