#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>
#include <tuple>
#include <chrono>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include <random>
#include "matrix.hpp"
#include "nn.hpp"

using namespace std;
using namespace std::chrono;

#define FILE_TRAIN_LABELS "../data/fashion_mnist_train_labels.csv"
#define FILE_TEST_LABELS "../data/fashion_mnist_test_labels.csv"
#define FILE_TRAIN_VECTORS "../data/fashion_mnist_train_vectors.csv"
#define FILE_TEST_VECTORS "../data/fashion_mnist_test_vectors.csv"
#define FILE_TRAIN_PREDICTIONS "../train_predictions.csv"
#define FILE_TEST_PREDICTIONS "../test_predictions.csv"

#define N_HIDDEN 64
#define EPOCHS 25
#define BATCH_SIZE 32
#define LEARNING_RATE 0.5
#define NORMALIZE_DATA true
#define TEST_ACCURACY true
#define WRITE_OUTPUT true

vector<int> read_labels(const string file_path) {
    std::ifstream file(file_path);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    vector<int> labels;
    string line;

    while (getline(file, line)) {
        try {
            labels.push_back(stoi(line));
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Invalid line: " + line);
        }
    }

    file.close();
    return labels;
}

vector<vector<float>> read_vectors(string file_path) {
    std::ifstream file(file_path);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    vector<vector<float>> vectors;
    string line;
    while (getline(file, line))
    {
        vector<float> row;
        std::stringstream line_stream(line);
        string pixel;
        int pixel_count = 0;

        while (getline(line_stream, pixel, ','))
        {
            try {
                int value = std::stoi(pixel);
                row.push_back(value);
                pixel_count++;
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("Invalid numeric value: " + pixel);
            }
        }

        vectors.push_back(row);
    }

    file.close();
    return vectors;
}

void write_predictions(string file_path, vector<int> predictions) {
    std::ofstream file(file_path); // Open the file for writing
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    for (int pred : predictions) {
        file << pred << endl;
    }

    file.close();
}

void normalize(vector<vector<float>> &vectors)
{
    for (auto& row : vectors) {
        for (auto& value : row) {
            value = value / 255.0f;
        }
    }
}

Network init_network() {
    std::srand(std::time(0));

    Layer hidden_layer = {
        784,
        N_HIDDEN,
        Matrix(784, N_HIDDEN),
        Matrix(1, N_HIDDEN),
        Matrix(1, N_HIDDEN)
    };

    std::default_random_engine generator(std::time(0));
    std::normal_distribution<float> he(0, 2.0f / 784); // normal He

    for (int i = 0; i < 784; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            hidden_layer.weights.set(i, j, he(generator));
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        hidden_layer.biases.set(0, j, he(generator));
    }

    Layer output_layer = {
        N_HIDDEN,
        10,
        Matrix(N_HIDDEN, 10),
        Matrix(1, 10),
        Matrix(1, 10)
    };

    std::normal_distribution<float> glorot(0, 2.0f / (N_HIDDEN + 10)); // normal Glorot

    for (int i = 0; i < N_HIDDEN; i++) {
        for (int j = 0; j < 10; ++j) {
            output_layer.weights.set(i, j, glorot(generator));
        }
    }
    for (int j = 0; j < 10; j++) { 
        output_layer.biases.set(0, j, glorot(generator));
    }

    Network network = {
        hidden_layer,
        output_layer
    };

    return network;
}

int main()
{
    printf("No. of hidden neurons: %d\n", N_HIDDEN);
    printf("Learning rate:         %.3f\n", LEARNING_RATE);
    printf("Batch size:            %d\n\n", BATCH_SIZE);

    auto start = high_resolution_clock::now();

    vector<int> train_labels = read_labels(FILE_TRAIN_LABELS);
    vector<int> test_labels = read_labels(FILE_TEST_LABELS);
    vector<vector<float>> train_vectors = read_vectors(FILE_TRAIN_VECTORS);
    vector<vector<float>> test_vectors = read_vectors(FILE_TEST_VECTORS);

    if (NORMALIZE_DATA) {
        normalize(train_vectors);
        normalize(test_vectors);
    }

    Network nn = init_network();
    train(nn, EPOCHS, BATCH_SIZE, LEARNING_RATE, TEST_ACCURACY, train_vectors, train_labels, test_vectors, test_labels);

    if (WRITE_OUTPUT) {
        int correct = 0;
        vector<int> train_predictions;
        for (size_t i = 0; i < train_vectors.size(); i++) {
            int prediction = predict(nn, train_vectors[i]);
            train_predictions.push_back(prediction);
            correct += prediction == train_labels[i] ? 1 : 0;
        }

        vector<int> test_predictions;
        for (size_t i = 0; i < test_vectors.size(); i++) {
            test_predictions.push_back(predict(nn, test_vectors[i]));
        }

        printf("Train prediction: %d/50000 ~ %.2f\n", correct, correct / 50000.0f * 100);

        write_predictions(FILE_TRAIN_PREDICTIONS, train_predictions);
        write_predictions(FILE_TEST_PREDICTIONS, test_predictions);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    int seconds = duration.count() / 1000.0f;
    cout << "Total time: " << seconds / 60 << " min " << seconds % 60 << " s\n";
}