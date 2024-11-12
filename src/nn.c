#include <math.h>
#include <stdlib.h>
#include "nn.h"

#define N_INPUT 784
#define N_HIDDEN 512
#define N_OUTPUT 10
#define LEARNING_RATE 0.1
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

/// @brief Fisher-Yates array shuffle algorithm
/// @param data Data array
/// @param n Length of array
void shuffle(int **data, int n) {
    srand(42); // dont call seed every time?
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int *temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

float relu(float x) {
    return fmax(x, 0);
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

int forward(int **input) {
    int result = 0;
    return result;
}