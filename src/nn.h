typedef struct Layer {
    int n_input, n_output;
    float *weights;
    float *biases;
} Layer;

typedef struct Network {
    Layer hidden;
    Layer output;
} Network;

void shuffle(int **data, int n);
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_derivative(float x);
int forward(int **input);