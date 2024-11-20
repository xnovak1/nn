#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
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
 */
void softmax(vector<float> &input) {
    float sum = 0, max = input[0];

    for (float num : input) {
        if (num > max)
            max = num;
    }

    for (size_t i = 0; i < input.size(); i++) {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }

    for (size_t i = 0; i < input.size(); i++)
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
void forward(Layer &layer, vector<float> input, vector<float> &output, bool activate) {
    for (int j = 0; j < layer.n_output; j++) {
        output[j] += layer.biases.at(0, j);
        for (int i = 0; i < layer.n_input; i++) {
            output[j] += input[i] * layer.weights.at(i, j);
        }

        if (activate) output[j] = activation(output[j]);
    }
}

/**
 * @brief Prediction of single image.
 * 
 * @param input Vector of pixels
 * @return int Image classification (0-9)
 */
int predict(Network network, vector<float> input) {
    vector<float> inputF(input.begin(), input.end());
    vector<float> output_hidden(network.hidden.n_output, 0);
    vector<float> output_last(network.output.n_output, 0);

    forward(network.hidden, inputF, output_hidden, true);
    forward(network.output, output_hidden, output_last, false);
    softmax(output_last);

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
    for (int j = 0; j < predicted.size(); j++)
    {
       for (int i = 0; i < 10; i++)
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
    for (size_t i = 0; i < gradient.rowSize(); i++)
    {
        for (size_t j = 0; j < gradient.colSize(); j++)
        {
            gradient.set(i,j,relu_derivative(gradient.at(i,j)));
        }
    }

    //update weights and biases for hidden layer
    nn.hidden.weights = nn.hidden.outputs.transpose()*gradient;

    nn.hidden.biases = gradient.SumRowsToOne();
}

/**
 * @brief Trains the network using current minibatch.
 * 
 * @param nn Neural network
 * @param minibatch Current minibatch
 */
void train_batch(Network &nn, vector<tuple<vector<float>, float>> minibatch) {
    ;
}

/**
 * @brief Trains neural network on supplied data and tests it against test data.
 * 
 * @param nn Neural network
 * @param epochs Number of epochs
 * @param batch_size Minibatch size
 * @param train_vectors Training data
 * @param train_labels Training data
 * @param test_vectors Testing data
 * @param test_labels Testing data
 */
void train(
    Network &nn,
    int epochs,
    int batch_size,
    bool test_accuracy,
    vector<vector<float>> train_vectors,
    vector<int> train_labels,
    vector<vector<float>> test_vectors,
    vector<int> test_labels) {

    auto rng = std::default_random_engine {};

    // we create vector of tuples because of shuffling of data each epoch
    vector<tuple<vector<float>, int>> input;
    for (size_t i = 0; i < train_vectors.size(); i++) {
        input.emplace_back(train_vectors[i], train_labels[i]);
    }

    for (int epoch; epoch < epochs; epoch++) {
        std::shuffle(input.begin(), input.end(), rng);
        for (size_t i = 0; i < input.size() / batch_size; i ++) {
            vector<tuple<vector<float>, float>> minibatch(batch_size);
            for (int j = 0; j < batch_size; j++) {
                minibatch[j] = input[i * batch_size + j];
            }

            train_batch(nn, minibatch);
        }

        if (test_accuracy) {
            int correct = 0;
            for (size_t k = 0; k < test_vectors.size(); k++) {
                int predicted = predict(nn, test_vectors[k]);
                correct += predicted == test_labels[k] ? 1 : 0;
            }
            float accuracy = (float)correct / test_labels.size();
            printf("Epoch %2.d: %d/10000 ~ %.2f %%\n", epoch + 1, correct, accuracy * 100);
        }
    }
}