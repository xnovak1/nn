#define N_INPUT 784
#define N_HIDDEN 100
#define N_OUTPUT 10
#define LEARNING_RATE 0.1
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

typedef struct Layer {
    float *weights;
} Layer;

typedef struct Network {
    Layer input;
    Layer hidden;
    Layer output;
} Network;