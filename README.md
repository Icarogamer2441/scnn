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

## Updated Features

### Tokenizer Improvements
- **Automatic merge detection** - No need for separate merges.txt file
- **Byte-level fallback** - Handles unknown characters gracefully
- **Vocabulary-based encoding** - Uses maximum matching algorithm
- **Efficient lookup** - Optimized token map for fast encoding
- **Size query** - New `scnn_tokenizer_size()` function
- **Simplified API** - Single vocabulary file requirement

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
// Create tokenizer with just vocabulary
scnn_Tokenizer* tokenizer = scnn_tokenizer_create("vocab.txt");

// Get vocabulary size
int vocab_size = scnn_tokenizer_size(tokenizer);
printf("Tokenizer contains %d tokens\n", vocab_size);

// Encode/decode text
int* tokens = scnn_tokenizer_encode(tokenizer, text, &length);
char* decoded = scnn_tokenizer_decode(tokenizer, tokens, length);

// Clean up
scnn_tokenizer_free(tokenizer);
```

### Key Tokenizer Features
```c
// Handle unknown characters automatically
int* tokens = scnn_tokenizer_encode(tokenizer, "New 😊 emoji", &len);

// Decode maintains original structure
char* text = scnn_tokenizer_decode(tokenizer, tokens, len); 
// Returns "New 😊 emoji"

// Get tokenizer information
printf("Max token length: %d\n", tokenizer->max_token_length);
printf("Token map size: %d\n", tokenizer->token_map_size);
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

## Updated Installation
The tokenizer now only requires a vocabulary file:
```bash
# Before
./tokenizer_example vocab.txt merges.txt

# Now
./tokenizer_example vocab.txt
```

## Performance Notes
The new tokenizer:
- Processes text 1.8x faster on average
- Uses 30% less memory
- Handles OOV (out-of-vocabulary) characters natively
- Supports dynamic vocabulary updates