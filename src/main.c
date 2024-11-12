#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

#define TRAIN_VECTORS_FILE "data/fashion_mnist_train_vectors.csv"
#define TRAIN_LABELS_FILE "data/fashion_mnist_train_labels.csv"
#define TEST_VECTORS_FILE "data/fashion_mnist_test_vectors.csv"
#define TEST_LABELS_FILE "data/fashion_mnist_test_labels.csv"

#define TRAIN_PREDICTIONS_FILE "train_predictions.csv"
#define TEST_PREDICTIONS_FILE "test_predictions.csv"

#define N_TRAIN_SAMPLES 60000
#define N_TEST_SAMPLES 10000


int read_input(char *filename, int n_cols, int n_rows, int ***output) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Unable to open file!");
        printf("%s\n", filename);
        return 1;
    }

    int **values = malloc(n_rows * sizeof(int *));
    if (values == NULL) {
        perror("Memory allocation failed!");
        fclose(file);
        return 1;
    }

    char line[4096];
    int row_count = 0;

    while (fgets(line, sizeof(line), file)) {
        values[row_count] = malloc(n_cols * sizeof(int));
        if (values[row_count] == NULL) {
            perror("Memory allocation failed!");
            fclose(file);
            return 1;
        }

        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < n_cols) {
            values[row_count][col] = atoi(token);
            token = strtok(NULL, ",");
            col++;
        }

        row_count++;
    }

    fclose(file);
    *output = values;
    return 0;
}

int write_output();

int main(int argc, char **argv) {
    int ret_val = 0;

    int **train_vectors = NULL;
    int **train_labels = NULL;
    int **test_vectors = NULL;
    int **test_labels = NULL;

    if (read_input(TRAIN_VECTORS_FILE, 784, N_TRAIN_SAMPLES, &train_vectors) != 0 ||
        read_input(TRAIN_LABELS_FILE, 1, N_TRAIN_SAMPLES, &train_labels) != 0 ||
        read_input(TEST_VECTORS_FILE, 784, N_TEST_SAMPLES, &test_vectors) != 0 ||
        read_input(TEST_LABELS_FILE, 1, N_TEST_SAMPLES, &test_labels) != 0) {
        ret_val = 1;
        goto cleanup;
    }

    // training and testing network...

cleanup:
    if (train_vectors != NULL) {
        for (int i = 0; i < N_TRAIN_SAMPLES; i++)
            free(train_vectors[i]);
        free(train_vectors);
    }
    
    if (train_labels != NULL) {
        for (int i = 0; i < N_TRAIN_SAMPLES; i++)
            free(train_labels[i]);
        free(train_labels);
    }

    if (test_vectors != NULL) {
        for (int i = 0; i < N_TEST_SAMPLES; i++)
            free(test_vectors[i]);
        free(test_vectors);
    }

    if (test_labels != NULL) {
        for (int i = 0; i < N_TEST_SAMPLES; i++)
            free(test_labels[i]);
        free(test_labels);
    }

    return ret_val;
}