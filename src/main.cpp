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

#define N_PIXELS 28*28
#define N_HIDDEN 16
#define EPOCHS 10
#define BATCH_SIZE 32
#define LEARNING_RATE 0.1
#define NORMALIZE_DATA false
#define TEST_ACCURACY true
#define WRITE_OUTPUT false

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

vector<vector<float>> read_vectors(
    string file_path,
    bool statistics,
    float &all_values,
    float &all_values_sq)
{
    std::ifstream file(file_path);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    vector<vector<float>> vectors;
    all_values = 0;
    all_values_sq = 0;

    string line;
    while (getline(file, line))
    {
        vector<float> row;
        int row_sum = 0;
        int row_sum_sq = 0;

        std::stringstream line_stream(line);
        string pixel;
        int pixel_count = 0;

        while (getline(line_stream, pixel, ','))
        {
            try {
                int value = std::stoi(pixel);
                row.push_back(value);

                if (statistics) {
                    row_sum += value;
                    row_sum_sq += value * value;
                }

                pixel_count++;
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("Invalid numeric value: " + pixel);
            }
        }

        vectors.push_back(row);

        if (statistics) {
            all_values += row_sum / N_PIXELS;
            all_values_sq += row_sum_sq / N_PIXELS;
        }
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

void normalize(vector<vector<float>> &vectors, float mean, float sd)
{
    if (sd == 0.0f) {
        throw std::invalid_argument("Standard deviation can't be zero.");
    }

    for (auto& row : vectors) {
        for (auto& value : row) {
            value = (value - mean) / sd;
        }
    }
}

Network init_network() {
    std::srand(std::time(0));

    Layer hidden_layer = {
        N_PIXELS,
        N_HIDDEN,
        Matrix(N_PIXELS, N_HIDDEN),
        Matrix(1, N_HIDDEN),
        Matrix(1, N_HIDDEN)
    };

    std::default_random_engine generator(std::time(0));
    std::normal_distribution<float> he(0, 2.0f / N_PIXELS); // normal He

    for (int i = 0; i < N_PIXELS; i++) {
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
    auto start = high_resolution_clock::now();

    float all_values = 0;
    float all_values_sq = 0;

    // load data
    vector<int> train_labels = read_labels(FILE_TRAIN_LABELS);
    vector<int> test_labels = read_labels(FILE_TEST_LABELS);
    vector<vector<float>> train_vectors = read_vectors(FILE_TRAIN_VECTORS, true, all_values, all_values_sq);
    vector<vector<float>> test_vectors = read_vectors(FILE_TEST_VECTORS, false, all_values, all_values_sq);

    // normalize data
    if (NORMALIZE_DATA) {
        int num_of_images = (train_vectors.size());
        float mean = all_values / num_of_images;
        float sd = sqrt(all_values_sq / num_of_images - mean * mean);
        normalize(train_vectors, mean, sd);
        normalize(test_vectors, mean, sd);
    }

    Network nn = init_network();
    train(nn, EPOCHS, BATCH_SIZE, LEARNING_RATE, TEST_ACCURACY, train_vectors, train_labels, test_vectors, test_labels);

    if (WRITE_OUTPUT) {
        vector<int> train_predictions;
        for (size_t i = 0; i < train_vectors.size(); i++) {
            train_predictions.push_back(predict(nn, train_vectors[i]));
        }

        vector<int> test_predictions;
        for (size_t i = 0; i < test_vectors.size(); i++) {
            test_predictions.push_back(predict(nn, test_vectors[i]));
        }

        write_predictions(FILE_TRAIN_PREDICTIONS, train_predictions);
        write_predictions(FILE_TEST_PREDICTIONS, test_predictions);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Total time: " << duration.count() / 1000.0f << " seconds" << endl;
}