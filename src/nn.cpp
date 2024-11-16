#include <cmath>
#include <vector>
#include <algorithm>
#include "matrix.hpp"
#include "nn.hpp"

#include <iostream> // remove

using namespace std;

#define N_INPUT 784
#define N_HIDDEN 512
#define N_OUTPUT 10
#define LEARNING_RATE 0.1
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

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
    vector<float> output_hidden(network.hidden.n_output, 0);
    vector<float> output_last(network.output.n_output, 0);

    layer_forward(network.hidden, input, output_hidden, true);
    layer_forward(network.output, output_hidden, output_last, false);
    softmax(output_last, 10);

    return argmax(output_last);
}

/**
 * @brief Cross entropy for final
 * 
 * @param actual actual categories
 * @param predicted vector of predicted probabilities
 * @return cross entropy
 */
float cross_entropy(vector<int> actual, vector<vector<float>> predicted){
    float entropy;
    for (size_t j = 0; j < predicted.size(); j++)
    {
       for (size_t i = 0; i < 10; i++)
        {
            if (i==actual[j])
            {
                entropy += log(predicted[j][i]);
            }
        } 
    }
    return -(1/predicted.size())*entropy;
}

void backprop(Matrix labels, Network nn){
    //Softmax layer
    //predicted values - ground truth(labels)
    Matrix gradient = nn.output.outputs;
    for (size_t i = 0; i < labels.rowSize(); i++)
    {
        gradient.set(i,labels.at(i,0),gradient.at(i,labels.at(i,0))-1);
    }
    gradient = gradient*(1/labels.rowSize()); //normalize by number of inputs

    //update weights and biases of output layer
    gradient = gradient*nn.output.weights.transpose();

    nn.output.weights = nn.output.outputs.transpose()*gradient;

    nn.output.biases = gradient.SumRowsToOne();

    //Backprop ReLu gradient
    for (int i = 0; i < gradient.rowSize(); i++)
    {
        for (int j = 0; j < gradient.colSize(); j++)
        {
            gradient.set(i,j,relu_derivative(gradient.at(i,j)));
        }
    }

    //update weights and biases for hidden layer
    nn.hidden.weights = nn.hidden.outputs.transpose()*gradient;

    nn.hidden.biases = gradient.SumRowsToOne();
}