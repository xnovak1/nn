#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include "matrix.hpp"
#include "nn.hpp"

#include <iostream> // remove

using namespace std;

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
    vector<float> output_hidden(network.hidden.n_output, 0);
    vector<float> output_last(network.output.n_output, 0);

    forward(network.hidden, input, output_hidden, true);
    forward(network.output, output_hidden, output_last, false);
    softmax(output_last);

    return argmax(output_last);
}

void update_velocity(Network &nn, float momentum, const Matrix &hidden_grad, const Matrix &output_grad) {
    for (int i = 0; i < nn.hidden.weights_velocity.rowSize(); i++) {
        for (int j = 0; j < nn.hidden.weights_velocity.colSize(); j++) {
            float new_velocity = momentum * nn.hidden.weights_velocity.at(i, j) + (1 - momentum) * hidden_grad.at(i, j);
            nn.hidden.weights_velocity.set(i, j, new_velocity);
        }
    }

    for (int i = 0; i < nn.output.weights_velocity.rowSize(); i++) {
        for (int j = 0; j < nn.output.weights_velocity.colSize(); j++) {
            float new_velocity = momentum * nn.output.weights_velocity.at(i, j) + (1 - momentum) * output_grad.at(i, j);
            nn.output.weights_velocity.set(i, j, new_velocity);
        }
    }

    for (int i = 0; i < nn.hidden.biases_velocity.rowSize(); i++) {
        for (int j = 0; j < nn.hidden.biases_velocity.colSize(); j++) {
            float new_velocity = momentum * nn.hidden.biases_velocity.at(i, j) + (1 - momentum) * hidden_grad.at(i, j);
            nn.hidden.biases_velocity.set(i, j, new_velocity);
        }
    }

    for (int i = 0; i < nn.output.biases_velocity.rowSize(); i++) {
        for (int j = 0; j < nn.output.biases_velocity.colSize(); j++) {
            float new_velocity = momentum * nn.output.biases_velocity.at(i, j) + (1 - momentum) * output_grad.at(i, j);
            nn.output.biases_velocity.set(i, j, new_velocity);
        }
    }
}

/**
 * @brief Trains the network using current minibatch.
 * 
 * @param nn Neural network
 * @param minibatch Current minibatch
 * @param learning_rate Learning rate
 */
void train_batch(
    Network &nn,
    vector<tuple<vector<float>, float>> minibatch,
    float learning_rate,
    float momentum) {

    Matrix output_grad(nn.output.n_input, nn.output.n_output);
    Matrix hidden_grad(nn.hidden.n_input, nn.hidden.n_output);

    for (auto input : minibatch) {
        Matrix partial_output_grad(nn.output.n_input, nn.output.n_output);
        Matrix partial_hidden_grad(nn.hidden.n_input, nn.hidden.n_output);
        vector<float> output_hidden(nn.hidden.n_output, 0);
        vector<float> output_last(nn.output.n_output, 0);

        forward(nn.hidden, get<0>(input), output_hidden, true);
        forward(nn.output, output_hidden, output_last, false);
        softmax(output_last);

        float label = get<1>(input);

        // Output layer error
        vector<float> delta_output(nn.output.n_output, 0);
        for (int i = 0; i < nn.output.n_output; i++) {
            delta_output[i] = output_last[i] - (i == label ? 1.0f : 0.0f);
        }

        // Output layer gradient
        for (int i = 0; i < nn.output.n_input; i++) {
            for (int j = 0; j < nn.output.n_output; j++) {
                partial_output_grad.set(i, j, delta_output[j] * output_hidden[i]);
            }
        }

        // Hidden layer error
        vector<float> delta_hidden(nn.hidden.n_output, 0);
        for (int i = 0; i < nn.hidden.n_output; i++) {
            float error = 0;
            for (int j = 0; j < nn.output.n_output; j++) {
                error += delta_output[j] * nn.output.weights.at(i, j);
            }
            delta_hidden[i] = error * relu_derivative(output_hidden[i]);
        }

        // Hidden layer gradient
        for (int i = 0; i < nn.hidden.n_input; i++) {
            for (int j = 0; j < nn.hidden.n_output; j++) {
                partial_hidden_grad.set(i, j, delta_hidden[j] * get<0>(input)[i]);
            }
        }

        output_grad += partial_output_grad;
        hidden_grad += partial_hidden_grad;
    }

    // update parameter velocity
    update_velocity(nn, momentum, hidden_grad, output_grad);

    // update weights + biases
    nn.output.weights -= nn.output.weights_velocity * (learning_rate / minibatch.size());
    nn.output.biases -= nn.output.biases_velocity * (learning_rate / minibatch.size());
    nn.hidden.weights -= nn.hidden.weights_velocity * (learning_rate / minibatch.size());
    nn.hidden.biases -= nn.hidden.biases_velocity * (learning_rate / minibatch.size());
}

/**
 * @brief Trains neural network on supplied data and tests it against test data.
 * 
 * @param nn Neural network
 * @param epochs Number of epochs
 * @param batch_size Minibatch size
 * @param learning_rate Learning rate
 * @param train_vectors Training data
 * @param train_labels Training data
 * @param validation_vectors Testing data
 * @param validation_labels Testing data
 */
void train(
    Network &nn,
    int epochs,
    int batch_size,
    float learning_rate,
    float momentum,
    bool test_accuracy,
    vector<vector<float>> train_vectors,
    vector<int> train_labels,
    vector<vector<float>> validation_vectors,
    vector<int> validation_labels) {

    auto rng = std::default_random_engine {};

    // we create vector of tuples because of shuffling of data each epoch
    vector<tuple<vector<float>, int>> input;
    for (size_t i = 0; i < train_vectors.size(); i++) {
        input.emplace_back(train_vectors[i], train_labels[i]);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        // learning rate dropping
        if (epoch > 4) learning_rate = 0.005;
        if (epoch > 7) learning_rate = 0.001;

        std::shuffle(input.begin(), input.end(), rng);
        for (size_t i = 0; i < input.size() / batch_size; i ++) {
            vector<tuple<vector<float>, float>> minibatch(batch_size);
            for (int j = 0; j < batch_size; j++) {
                minibatch[j] = input[i * batch_size + j];
            }

            train_batch(nn, minibatch, learning_rate, momentum);
        }

        if (test_accuracy) {
            int correct = 0;
            for (size_t k = 0; k < validation_vectors.size(); k++) {
                int predicted = predict(nn, validation_vectors[k]);
                correct += predicted == validation_labels[k] ? 1 : 0;
            }
            float accuracy = (float)correct / validation_labels.size();
            printf("Epoch %2.d: %d/%ld ~ %.2f %%\n", epoch + 1, correct, validation_labels.size(), accuracy * 100);
        }
    }
}