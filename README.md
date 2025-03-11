# Simple C Neural Network (SCNN)

SCNN is a lightweight neural network library implemented in C, designed for educational purposes and simple machine learning tasks.

## Features

- Basic feedforward neural network implementation
- Support for multiple hidden layers
- Sigmoid activation function
- Backpropagation training
- Temperature-controlled predictions
- Model saving and loading
- Simple tokenizer implementation
- Easy-to-use API
- Minimal dependencies

## Installation

### Building from source

1. Clone the repository:
   ```bash
   git clone https://github.com/Icarogamer2441/scnn.git
   cd scnn
   ```

2. Build the library:
   ```bash
   make
   ```

3. (Optional) Install system-wide:
   ```bash
   sudo make install
   ```

### Using the library

Include the header in your C program:
```c
#include <scnn/scnn.h>
```

Link the library when compiling:
```bash
gcc -lscnn -o my_program my_program.c
```

## Usage

### Creating a Neural Network

```c
int hidden_sizes[] = {4, 3}; // Two hidden layers with 4 and 3 neurons
scnn_NeuralNetwork* net = scnn_create(2, 2, hidden_sizes, 1);
```

### Training the Network

```c
scnn_train(net, input, target, learning_rate);
```

### Making Predictions

```c
// Normal prediction
double* output = scnn_predict(net, input);

// Temperature-controlled prediction
double* temp_output = scnn_predict_temp(net, input, 0.5);
```

### Saving and Loading Models

```c
// Save model
scnn_save_model(net, "model.bin");

// Load model
scnn_NeuralNetwork* loaded_net = scnn_load_model("model.bin");
```

### Using the Tokenizer

```c
scnn_Tokenizer* tokenizer = scnn_tokenizer_create("vocab.txt", "merges.txt");
int* tokens = scnn_tokenizer_encode(tokenizer, "Hello world!", &length);
char* text = scnn_tokenizer_decode(tokenizer, tokens, length);
```

## Examples

The `examples/` directory contains sample programs demonstrating how to use the library:

- `basic_nn.c`: Simple neural network example with multiple hidden layers
- `tokenizer_example.c`: Basic tokenizer usage

To build the examples:
```bash
cd examples
make
```

## Documentation

The API is documented in the header file `include/scnn.h`.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.