# nn
Implementation of a MLP trained on Fashion-MNIST.

## Architecture

- Two-layer neural network (784-64-10)
- Activation function (hidden layer): ReLU
- Activation function (output layer): Softmax
- Loss function: Cross-entropy
- Optimizer: SGD + Momentum

## Performance

```bash
Training Dataset:      FashionMNIST
No. of hidden neurons: 64
Learning rate:         0.100
Momentum:              0.8
Batch size:            32

Epoch  1: 8407/10000 ~ 84.07 %
Epoch  2: 8325/10000 ~ 83.25 %
Epoch  3: 8648/10000 ~ 86.48 %
Epoch  4: 8770/10000 ~ 87.70 %
Epoch  5: 8701/10000 ~ 87.01 %
Epoch  6: 8829/10000 ~ 88.29 %
Epoch  7: 8840/10000 ~ 88.40 %
Epoch  8: 8846/10000 ~ 88.46 %
Epoch  9: 8847/10000 ~ 88.47 %
Epoch 10: 8848/10000 ~ 88.48 %
```

## Usage

1. Download the Fashion MNIST dataset and save it to data folder.
2. Change current directory to src:
   ```bash
   cd src
   ```
3. Compile the program:
   ```bash
   g++ -Wall -O3 main.cpp nn.cpp matrix.cpp -o network
   ```
4. Run the executable:
   ```bash
   ./network
   ```

The program will train the network on the Fashion MNIST dataset and output the accuracy for each epoch.
Using preprocessor macros, you can customize number of hidden neurons, learning rate, momentum, batch size, number of epochs.
You can also train the network on the original MNIST dataset (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), but you must use macro #define DATASET "MNIST".
