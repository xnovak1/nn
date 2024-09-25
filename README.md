# nn
Implementation of a MLP trained on Fashion-MNIST.

## Architecture

- Two-layer neural network (784-100-10)
- Activation function (hidden layer): ReLU
- Activation function (output layer): Softmax
- Loss function: Cross-entropy
- Optimizer: Adam

## Performance

```bash
Epoch  1, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  2, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  3, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  4, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  5, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  6, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  7, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  8, Accuracy: 0.00%, Avg Loss: 0.000
Epoch  9, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 10, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 11, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 12, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 13, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 14, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 15, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 16, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 17, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 18, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 19, Accuracy: 0.00%, Avg Loss: 0.000
Epoch 20, Accuracy: 0.00%, Avg Loss: 0.000
```

## Usage

1. Download the Fashion MNIST dataset.
2. Compile the program:
   ```bash
   gcc -O3 -o nn nn.c
   ```
3. Run the executable:
   ```bash
   ./nn
   ```

The program will train the network on the Fashion MNIST dataset and output the accuracy and average loss for each epoch.
